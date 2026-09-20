"""FedProto 算法实现。"""

import torch

from ..core import BaseClient, BaseServer
from ..gfl.fedproc import aggregate_prototypes, extract_prototypes


def add_arguments(parser):
    parser.add_argument("--mu", type=float, default=1.0)


class FedProtoClient(BaseClient):
    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.mu: float = args.mu

    def train(self, task):
        prototypes = task.payload.get("global_prototypes")
        if prototypes is not None:
            prototypes = prototypes.to(self.device)
        optimizer = self.build_optimizer()
        self.model.train()
        loss_sum, steps = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                features = self.model.extractor(inputs)
                loss = self.ce_loss(self.model.classifier(features), targets)
                if prototypes is not None:
                    loss = loss + self.mu * torch.nn.functional.mse_loss(
                        features, prototypes[targets]
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
        local, counts = extract_prototypes(
            self.model,
            self.train_loader(task.client_id),
            self.num_classes,
            self.feature_dim,
            self.device,
        )
        return self.result(
            task, loss_sum / steps, {"prototypes": local, "counts": counts}
        )


class Server(BaseServer):
    client_class = FedProtoClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        self.global_prototypes = None
        self.mu: float = args.mu
        self.feature_dim: int = args.feature_dim

    def training_payloads(self):
        return {
            client_id: {"global_prototypes": self.global_prototypes}
            for client_id in self.selected
        }

    def apply_results(self, results):
        self.update_client_states(results)
        self.global_prototypes = aggregate_prototypes(
            [results[client_id].payload for client_id in self.selected],
            self.global_prototypes
            if self.global_prototypes is not None
            else torch.zeros(self.num_classes, self.feature_dim),
            by_count=True,
        )

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "global_prototypes": self.global_prototypes,
        }
