"""FedTGP 算法实现。"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..core import BaseClient, BaseServer
from ..gfl.fedpln import dist_contrastive_loss
from ..gfl.fedproc import aggregate_prototypes, extract_prototypes


def add_arguments(parser):
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--server-epochs", type=int, default=10)
    parser.add_argument("--server-lr", type=float, default=0.01)
    parser.add_argument("--margin-threshold", type=float, default=10.0)


class TGP(torch.nn.Module):
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_classes, feature_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, classes):
        return self.net(self.embeddings(classes))


class FedTGPClient(BaseClient):
    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.lambda_: float = args.lambda_

    def train(self, task):
        prototypes = task.payload.get("prototypes")
        if prototypes is not None:
            prototypes = prototypes.to(self.device)
        optimizer = self.build_optimizer()
        ce_sum, proto_sum, batches = 0.0, 0.0, 0
        self.model.train()
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                features = self.model.extractor(inputs)
                ce = self.ce_loss(self.model.classifier(features), targets)
                proto = (
                    torch.zeros((), device=self.device)
                    if prototypes is None
                    else torch.nn.functional.mse_loss(features, prototypes[targets])
                )
                optimizer.zero_grad(set_to_none=True)
                (ce + self.lambda_ * proto).backward()
                optimizer.step()
                ce_sum += ce.item()
                proto_sum += proto.item()
                batches += 1
        local, counts = extract_prototypes(
            self.model,
            self.train_loader(task.client_id),
            self.num_classes,
            self.feature_dim,
            self.device,
        )
        return self.result(
            task,
            ce_sum / batches,
            {"prototypes": local, "counts": counts, "loss_proto": proto_sum / batches},
        )


class Server(BaseServer):
    client_class = FedTGPClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        self.tgp = TGP(self.num_classes, args.feature_dim).to(self.device)
        self.prototypes = None
        self.server_epochs: int = args.server_epochs
        self.server_lr: float = args.server_lr
        self.margin_threshold: float = args.margin_threshold
        self.feature_dim: int = args.feature_dim
        self.gap = torch.full((self.num_classes,), float("inf"), device=self.device)
        self.prototype_loss = 0.0

    def training_payloads(self):
        return {
            client_id: {"prototypes": self.prototypes} for client_id in self.selected
        }

    def apply_results(self, results):
        self.update_client_states(results)
        payloads = [results[client_id].payload for client_id in self.selected]
        old = (
            self.prototypes
            if self.prototypes is not None
            else torch.zeros(self.num_classes, self.feature_dim)
        )
        targets = aggregate_prototypes(payloads, old, by_count=True)
        distances = torch.cdist(targets.to(self.device), targets.to(self.device))
        distances.fill_diagonal_(float("inf"))
        self.gap = distances.min(dim=1).values
        valid = targets.norm(dim=1).to(self.device) > 1e-8
        self.gap[~valid] = self.gap[valid].min() if valid.any() else 0
        uploaded = [
            (row.to(self.device), label)
            for payload in payloads
            for label, row in enumerate(payload["prototypes"])
            if row.norm() > 1e-8
        ]
        if uploaded:
            optimizer = torch.optim.SGD(self.tgp.parameters(), lr=self.server_lr)
            values = torch.stack([row for row, _ in uploaded])
            labels = torch.tensor(
                [label for _, label in uploaded], device=self.device, dtype=torch.long
            )
            dataset = TensorDataset(values, labels)
            for _ in range(self.server_epochs):
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                for values, labels in loader:
                    values = values.to(self.device)
                    labels = labels.to(self.device, dtype=torch.long)
                    generated = self.tgp(
                        torch.arange(self.num_classes, device=self.device)
                    )
                    margin = min(self.gap.max().item(), self.margin_threshold)
                    loss = dist_contrastive_loss(values, generated, labels, margin)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    self.prototype_loss = loss.item()
            self.prototypes = (
                self.tgp(torch.arange(self.num_classes, device=self.device))
                .detach()
                .cpu()
            )
        else:
            self.prototypes = targets

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "tgp": self.tgp.state_dict(),
            "prototypes": self.prototypes,
            "gap": self.gap.cpu(),
        }
