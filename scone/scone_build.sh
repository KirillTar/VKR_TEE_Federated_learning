#!/usr/bin/env bash
# Build + sconify the FL server image for SGX deployment.
# No host-side `scone` CLI is required — uses the sconify-image Docker workflow.
# CAS: defaults to the public Community Edition CAS (scone-cas.cf:8081).
#
# Usage:
#   ./scone/scone_build.sh                # build base + sconify
#   ./scone/scone_build.sh --base         # build base image only (no sconify)
#   ./scone/scone_build.sh --hw-tolerant  # add tolerations needed on older
#                                           SGX1 CPUs (e.g. Xeon E-2386G,
#                                           E-2236, E3-1230v5) where Intel
#                                           has published TCB advisories.
#
# After build:
#   SIM mode (no real SGX, for testing):
#     docker run --rm -e SCONE_MODE=SIM fl-server:scone
#   HW mode (in-tree SGX driver, kernel >= 5.11):
#     docker run --rm -e SCONE_MODE=HW \
#       --device /dev/sgx_enclave --device /dev/sgx_provision \
#       fl-server:scone
#
# Override CMD at run time (e.g. measure_overhead.py instead of server.py):
#   docker run --rm -e SCONE_MODE=SIM fl-server:scone \
#     python3.11 /app/scone/measure_overhead.py --scenario cs --n_rounds 30

set -euo pipefail

IMAGE_BASE="fl-server:base"
IMAGE_SCONE="fl-server:scone"
SCONIFY_IMG="registry.scontain.com/sconectl/sconify-image:latest"

ONLY_BASE=false
HW_TOLERANT=false
for arg in "$@"; do
    case $arg in
        --base) ONLY_BASE=true ;;
        --hw-tolerant) HW_TOLERANT=true ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

echo "=== [1/2] Build base Docker image ($IMAGE_BASE) ==="
docker build -t "$IMAGE_BASE" -f scone/Dockerfile .

if [ "$ONLY_BASE" = true ]; then
    echo "Base built. Skipping sconification."
    exit 0
fi

# Tolerations: required on most non-latest CPUs because Intel SGX TCB advisories
# accumulate over time. Without these, attestation rejects the platform.
EXTRA_FLAGS=()
if [ "$HW_TOLERANT" = true ]; then
    EXTRA_FLAGS+=(--allow-tcb-vulnerabilities)
fi

echo "=== [2/2] Sconify via sconify-image (public CAS scone-cas.cf:8081) ==="
# Why these flags:
#   --base python:3.11-slim-bookworm
#                               same image fl-server:base extends from. Has
#                                 glibc 2.36 + system libs Python actually uses
#                                 (libsqlite3-0, libssl, libcrypto, ...). Plain
#                                 debian:12-slim lacks libsqlite3.so.0, which
#                                 makes `import flwr` fail (flwr → sqlite3 →
#                                 libsqlite3.so.0). Alpine's musl libc would be
#                                 incompatible.
#   --binary /usr/local/bin/python3.11
#                               the actual Python interpreter (not the
#                                 /usr/local/bin/python symlink, which is dropped
#                                 by sconification onto a debian base).
#   --plain /usr/local/lib/python3.11
#                               Python stdlib AND site-packages (pip installs
#                                 torch, flwr, opacus, numpy here). Without this,
#                                 sconified Python crashes with
#                                 "ModuleNotFoundError: encodings".
#                                 NOTE: using --plain (not --dir) because we use
#                                 --disable-session-upload below: encrypted dirs
#                                 (--dir) need an FSPF key delivered via CAS,
#                                 which we are skipping. --plain still runs
#                                 inside the enclave (memory protected), only
#                                 the on-disk image is unencrypted.
#                                 For confidentiality of code at rest, switch
#                                 to --dir + a working CAS session.
#   --plain /app                project code copied by Dockerfile (core/, scone/,
#                                 experiments/).
#   --host-path /etc/hosts /etc/resolv.conf
#                               needed for DNS resolution from inside the
#                                 enclave (Flower client/server gRPC).
#   --heap 2G                   enclave heap. Empirically SCONE SIM needs ~800 MB
#                                 just for imports + first multi_krum round
#                                 (host RSS was 350 MB; SCONE LibOS adds its own
#                                 overhead). 2G gives comfortable margin for
#                                 32-bit floats × n_clients × d. On HW with
#                                 EPC=512 MB this triggers paging — but heap
#                                 size itself can exceed EPC; SGX swaps pages
#                                 in/out (overhead, but works).
#   --dlopen 2                  allow unrestricted dynamic library loading
#                                 ("debug only" mode in SCONE). Required because
#                                 we use --plain for the Python install (see
#                                 above). With --dlopen 1 (authenticated), only
#                                 .so files from an encrypted FSPF region can
#                                 be loaded — that needs a working CAS session.
#   --cas-debug                 trust the public CE CAS (scone-cas.cf:8081) which
#                                 currently runs in debug mode. Remove this when
#                                 deploying against a production CAS.
#   --disable-session-upload    skip uploading the policy session to CAS. The
#                                 image is still encrypted, but no remote
#                                 attestation is possible. Acceptable for
#                                 overhead measurement; remove for production
#                                 attested deployment with own CAS.

docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    "$SCONIFY_IMG" \
    /usr/local/bin/sconify_image \
        --from "$IMAGE_BASE" \
        --to "$IMAGE_SCONE" \
        --base python:3.11-slim-bookworm \
        --binary /usr/local/bin/python3.11 \
        --plain /usr/local/lib/python3.11 \
        --plain /app \
        --host-path /etc/hosts \
        --host-path /etc/resolv.conf \
        --heap 2G \
        --dlopen 2 \
        --cas-debug \
        --disable-session-upload \
        "${EXTRA_FLAGS[@]}"

echo
echo "=== Done ==="
docker images --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}' \
    | grep -E '^(REPOSITORY|fl-server)' | head -5

cat <<EOF

Run examples:
  SIM mode:
    docker run --rm -e SCONE_MODE=SIM $IMAGE_SCONE
  HW mode (in-tree driver, kernel >= 5.11):
    docker run --rm -e SCONE_MODE=HW \\
      --device /dev/sgx_enclave --device /dev/sgx_provision \\
      $IMAGE_SCONE
  Override CMD (e.g. measurement script):
    docker run --rm -e SCONE_MODE=SIM $IMAGE_SCONE \\
      python3.11 /app/scone/measure_overhead.py --scenario cs --n_rounds 30
EOF
