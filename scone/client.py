"""
Flower FL client.
Designed to work both natively (GPU experiments) and inside SCONE (TEE demo).
"""

import argparse
import sys
import logging
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import get_model, get_weights, set_weights, weights_to_flat, flat_to_weights
from core.data import load_dataset, dirichlet_split, get_client_dataloader, get_test_dataloader
from core.metrics import compute_mta

logger = logging.getLogger("fl.client")
logging.basicConfig(level=logging.INFO)


class FLClient(fl.client.NumPyClient):

    def __init__(
        self,
        client_id: int,
        dataset: str,
        data_dir: str,
        n_clients: int,
        alpha: float,
        seed: int,
        local_epochs: int,
        batch_size: int,
        lr: float,
        momentum: float,
        weight_decay: float,
    ):
        self.client_id = client_id
        self.dataset_name = dataset
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data partition
        train_ds, test_ds = load_dataset(dataset, data_dir)
        client_indices = dirichlet_split(train_ds, n_clients, alpha, seed)

        self.train_loader = get_client_dataloader(train_ds, client_indices[client_id], batch_size)
        self.test_loader = get_test_dataloader(test_ds)

        self.model = get_model(dataset).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_parameters(self, config):
        return [w.numpy() for w in get_weights(self.model)]

    def set_parameters(self, parameters):
        weights = [torch.tensor(p) for p in parameters]
        set_weights(self.model, weights)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # Save initial weights to compute delta
        initial_weights = get_weights(self.model)

        self.model.train()
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                optimizer.step()

        # Send delta (trained - initial) to server
        trained_flat = weights_to_flat(get_weights(self.model))
        initial_flat = weights_to_flat(initial_weights)
        delta_flat = trained_flat - initial_flat
        delta_weights = flat_to_weights(delta_flat, initial_weights)

        return [w.numpy() for w in delta_weights], len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss_fn = nn.CrossEntropyLoss()

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                total_loss += loss_fn(logits, y).item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, total, {"accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--n_clients", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--server_address", default="127.0.0.1:8080")
    args = parser.parse_args()

    client = FLClient(
        client_id=args.client_id,
        dataset=args.dataset,
        data_dir=args.data_dir,
        n_clients=args.n_clients,
        alpha=args.alpha,
        seed=args.seed,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()
