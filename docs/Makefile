PANDOC = pandoc
REF = reference.docx
CHAPTERS_DIR = chapters
OUT_DIR = out

# All chapter .md files, sorted by name (use numeric prefixes: 01_, 02_, etc.)
CHAPTERS = $(sort $(wildcard $(CHAPTERS_DIR)/*.md))

# Individual chapter docx files
CHAPTER_DOCX = $(patsubst $(CHAPTERS_DIR)/%.md,$(OUT_DIR)/%.docx,$(CHAPTERS))

PANDOC_OPTS = --reference-doc=$(REF) \
              --from=markdown+pipe_tables+yaml_metadata_block+fenced_code_blocks+implicit_figures \
              --to=docx \
              --standalone \
              --number-sections

# Default: build all chapters as separate docx
all: $(CHAPTER_DOCX)

$(OUT_DIR)/%.docx: $(CHAPTERS_DIR)/%.md $(REF) | $(OUT_DIR)
	$(PANDOC) $(PANDOC_OPTS) -o $@ $<
	@echo "  Built $@"

# Build merged document from all chapters
merged: $(OUT_DIR)/vkr_full.docx

$(OUT_DIR)/vkr_full.docx: $(CHAPTERS) $(REF) | $(OUT_DIR)
	$(PANDOC) $(PANDOC_OPTS) -o $@ $(CHAPTERS)
	@echo "  Built $@ ($(words $(CHAPTERS)) chapters)"

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Build single chapter: make chapter NAME=01_intro
chapter:
	$(PANDOC) $(PANDOC_OPTS) -o $(OUT_DIR)/$(NAME).docx $(CHAPTERS_DIR)/$(NAME).md
	@echo "  Built $(OUT_DIR)/$(NAME).docx"

# Regenerate reference.docx
reference:
	cd .. && .venv/bin/python docs/make_reference.py

clean:
	rm -rf $(OUT_DIR)

list:
	@echo "Chapters found:"
	@for f in $(CHAPTERS); do echo "  $$f"; done

.PHONY: all merged chapter reference clean list
