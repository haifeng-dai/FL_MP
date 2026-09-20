"""FedSA 算法实现。"""

import torch

from ..core import BaseClient, BaseServer
from ..gfl.fedpln import dist_contrastive_loss
from ..gfl.fedproc import aggregate_prototypes, extract_prototypes


def add_arguments(parser):
    parser.add_argument("--alpha-sa", type=float, default=0.9)
    parser.add_argument("--lambda-r", type=float, default=1.0)
    parser.add_argument("--lambda-mcl", type=float, default=1.0)
    parser.add_argument("--lambda-cc", type=float, default=1.0)


def margin(anchors):
    valid = anchors[anchors.norm(dim=1) > 1e-8]
    if len(valid) < 2:
        return 0.0
    return torch.cdist(valid, valid).sum().item() / (len(valid) - 1) ** 2


class FedSAClient(BaseClient):
    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.lambda_r: float = args.lambda_r
        self.lambda_mcl: float = args.lambda_mcl
        self.lambda_cc: float = args.lambda_cc
        self.classes = torch.arange(self.num_classes, device=self.device)

    def train(self, task):
        anchors = task.payload["anchors"].to(self.device)
        previous = task.payload["previous"].to(self.device)
        threshold = max(margin(anchors), margin(previous))
        optimizer = self.build_optimizer()
        total, batches = 0.0, 0
        self.model.train()
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                features = self.model.extractor(inputs)
                loss = self.ce_loss(self.model.classifier(features), targets)
                loss = loss + self.lambda_r * torch.nn.functional.mse_loss(
                    features, anchors[targets]
                )
                loss = loss + self.lambda_mcl * dist_contrastive_loss(
                    features, anchors, targets, threshold
                )
                loss = loss + self.lambda_cc * self.ce_loss(
                    self.model.classifier(anchors), self.classes
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
        local, counts = extract_prototypes(
            self.model,
            self.train_loader(task.client_id),
            self.num_classes,
            self.feature_dim,
            self.device,
        )
        return self.result(
            task, total / batches, {"prototypes": local, "counts": counts}
        )


class Server(BaseServer):
    client_class = FedSAClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        self.alpha: float = args.alpha_sa
        self.anchors = torch.nn.functional.normalize(
            torch.randn(self.num_classes, args.feature_dim), dim=1
        )
        self.client_anchors = {i: self.anchors.clone() for i in range(self.num_clients)}

    def training_payloads(self):
        return {
            i: {"anchors": self.anchors, "previous": self.client_anchors[i]}
            for i in self.selected
        }

    def apply_results(self, results):
        self.update_client_states(results)
        self.aggregate_weighted(results)
        payloads = [results[i].payload for i in self.selected]
        for i in self.selected:
            self.client_anchors[i] = results[i].payload["prototypes"].clone()
        merged = aggregate_prototypes(payloads, self.anchors, by_count=True)
        mask = merged.norm(dim=1) > 1e-8
        self.anchors[mask] = (
            self.alpha * self.anchors[mask] + (1 - self.alpha) * merged[mask]
        )

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "anchors": self.anchors,
            "client_anchors": self.client_anchors,
        }
