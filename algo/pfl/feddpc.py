"""FedDPC 算法实现。"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..core import BaseClient, BaseServer, clone_state
from ..gfl.fedpln import PLN
from ..gfl.fedproc import extract_prototypes


def add_arguments(parser):
    parser.add_argument("--head-epochs", type=int, default=1)
    parser.add_argument("--body-epochs", type=int, default=1)
    parser.add_argument("--lr-head", type=float, default=0.01)
    parser.add_argument("--lr-body", type=float, default=0.01)
    parser.add_argument("--lambda-p", type=float, default=1.0)
    parser.add_argument("--lambda-acl", type=float, default=1.0)
    parser.add_argument("--server-epochs", type=int, default=10)
    parser.add_argument("--server-lr", type=float, default=0.01)


class FedDPCClient(BaseClient):
    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.head_epochs: int = args.head_epochs
        self.body_epochs: int = args.body_epochs
        self.lr_head: float = args.lr_head
        self.lr_body: float = args.lr_body
        self.lambda_p: float = args.lambda_p

    def train(self, task):
        prototypes = task.payload.get("prototypes")
        if prototypes is not None:
            prototypes = prototypes.to(self.device)
        loader = self.train_loader(task.client_id)
        self.model.train()
        self.model.extractor.requires_grad_(False)
        self.model.classifier.requires_grad_(True)
        head = torch.optim.SGD(
            self.model.classifier.parameters(),
            lr=self.lr_head,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        ce_sum = 0.0
        head_batches = 0
        labels = torch.arange(self.num_classes, device=self.device)
        for _ in range(self.head_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.ce_loss(self.model(x), y)
                if prototypes is not None:
                    loss = loss + self.lambda_p * self.ce_loss(
                        self.model.classifier(prototypes), labels
                    )
                head.zero_grad()
                loss.backward()
                head.step()
                ce_sum += loss.item()
                head_batches += 1
        self.model.extractor.requires_grad_(True)
        self.model.classifier.requires_grad_(False)
        proto_sum = 0.0
        body_batches = 0
        if prototypes is not None:
            body = torch.optim.SGD(
                self.model.extractor.parameters(),
                lr=self.lr_body,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            for _ in range(self.body_epochs):
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    loss = torch.nn.functional.mse_loss(
                        self.model.extractor(x), prototypes[y]
                    )
                    body.zero_grad()
                    loss.backward()
                    body.step()
                    proto_sum += loss.item()
                    body_batches += 1
        local, _ = extract_prototypes(
            self.model, loader, self.num_classes, self.feature_dim, self.device
        )
        return self.result(
            task,
            ce_sum / head_batches,
            {
                "prototypes": local,
                "loss_proto": proto_sum / body_batches if body_batches else 0.0,
            },
        )


class Server(BaseServer):
    client_class = FedDPCClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        self.pln = PLN(
            self.num_classes, args.feature_dim, args.feature_dim, 1, False, 0
        ).to(self.device)
        self.prototypes = None
        self.server_epochs: int = args.server_epochs
        self.server_lr: float = args.server_lr
        self.lambda_acl: float = args.lambda_acl

    def training_payloads(self):
        return {i: {"prototypes": self.prototypes} for i in self.selected}

    def apply_results(self, results):
        self.update_client_states(results)
        uploaded = [
            (row.to(self.device), label)
            for i in self.selected
            for label, row in enumerate(results[i].payload["prototypes"])
            if row.norm() > 1e-8
        ]
        if not uploaded:
            return
        optimizer = torch.optim.SGD(self.pln.parameters(), lr=self.server_lr)
        classes = torch.arange(self.num_classes, device=self.device)
        values = torch.stack([row for row, _ in uploaded])
        labels = torch.tensor(
            [label for _, label in uploaded], device=self.device, dtype=torch.long
        )
        loader = DataLoader(
            TensorDataset(values, labels), batch_size=self.batch_size, shuffle=True
        )
        for _ in range(self.server_epochs):
            for values, labels in loader:
                values = values.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)
                generated = self.pln(classes)
                orthogonal = torch.nn.functional.cross_entropy(
                    generated @ generated.T, classes
                )
                loss = torch.nn.functional.mse_loss(values, generated[labels])
                loss = loss + self.lambda_acl * orthogonal
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.prototypes = self.pln(classes).detach().cpu()

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "pln": clone_state(self.pln.state_dict()),
            "prototypes": self.prototypes,
        }
