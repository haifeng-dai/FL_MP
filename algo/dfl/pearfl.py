"""PearFL：基于邻居原型交换的去中心化个性化学习。"""

import torch

from ..core import BaseClient, BaseServer
from ..core.decentralized import adjacency, mix_states, sinkhorn
from ..gfl.fedproc import extract_prototypes


def add_arguments(parser):
    parser.add_argument("--lamda", type=float, default=1.0)


class PearFLClient(BaseClient):
    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.lamda: float = args.lamda

    def train(self, task):
        prototypes = task.payload["prototypes"].to(self.device)
        optimizer = self.build_optimizer()
        self.model.train()
        total, batches = 0.0, 0
        for inputs, targets in self.train_loader(task.client_id):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            features = self.model.extractor(inputs)
            logits = self.model.classifier(features)
            loss = self.ce_loss(logits, targets)
            if prototypes.abs().sum() > 0:
                loss = loss + self.lamda * torch.nn.functional.mse_loss(
                    features, prototypes[targets]
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
    client_class = PearFLClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        self.num_epochs = args.num_epochs
        self.weights = sinkhorn(adjacency(args)).to(self.device)
        shape = (self.num_clients, self.num_classes)
        self.local_prototypes = torch.zeros(*shape, args.feature_dim)
        self.local_counts = torch.zeros(*shape)
        self.personalized_prototypes = torch.zeros_like(self.local_prototypes)

    def training_payloads(self):
        return {
            client_id: {"prototypes": self.personalized_prototypes[client_id].clone()}
            for client_id in self.selected
        }

    def apply_results(self, results):
        self.update_client_states(results)
        for client_id in self.selected:
            payload = results[client_id].payload
            self.local_prototypes[client_id] = payload["prototypes"]
            self.local_counts[client_id] = payload["counts"]
        neighborhood = (self.weights > 0).float().cpu()
        weights = neighborhood[:, :, None] * self.local_counts[None, :, :]
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1)
        self.personalized_prototypes = torch.einsum(
            "ijc,jcd->icd", weights / denominator, self.local_prototypes
        )

    def mix_client_states(self):
        """在完成一轮内全部本地 epoch 后，执行一次模型邻居混合。"""
        self.client_states = {
            client_id: state
            for client_id, state in enumerate(
                mix_states(
                    [self.client_states[index] for index in range(self.num_clients)],
                    self.weights,
                    self.device,
                )
            )
        }

    def run_round(self):
        """PearFL 每轮在同一批客户端上交替执行本地单 epoch 与原型交换。"""
        self.select_clients()
        loss = 0.0
        for _ in range(self.num_epochs):
            results = self.run_clients()
            self.apply_results(results)
            loss = sum(results[client_id].loss for client_id in self.selected) / len(
                self.selected
            )
        self.mix_client_states()
        return loss, self.evaluate()

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "weights": self.weights.cpu(),
            "local_prototypes": self.local_prototypes,
            "local_counts": self.local_counts,
            "personalized_prototypes": self.personalized_prototypes,
        }
