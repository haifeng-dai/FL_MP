"""DFedSET：事件触发的模型与类别原型去中心化一致性训练。"""

import argparse

import torch

from algo.core import BaseClient, BaseServer
from algo.fedproc import extract_prototypes
from runtime import adjacency, metropolis_hastings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lambda-sa", type=float, default=0.0)
    parser.add_argument("--lambda-so", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument(
        "--confidence-mode", choices=("count", "log", "none"), default="log"
    )
    parser.add_argument(
        "--trigger-mode", choices=("adaptive", "global", "all"), default="adaptive"
    )
    parser.add_argument("--gamma-global", type=float, default=0.001)
    parser.add_argument("--no-relay", action="store_true")
    parser.add_argument("--no-redirect", action="store_true")


class DFedSETClient(BaseClient):
    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.lambda_sa: float = args.lambda_sa
        self.lambda_so: float = args.lambda_so
        self.confidence_mode: str = args.confidence_mode

    def train(self, task):
        consensus: torch.Tensor = task.payload["consensus"].to(self.device)
        optimizer = self.build_optimizer()
        self.model.train()
        total, batches = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                features = self.model.extractor(inputs)
                logits = self.model.classifier(features)
                loss = self.ce_loss(logits, targets)
                target = consensus[targets]
                if self.lambda_sa:
                    loss = loss + self.lambda_sa * torch.nn.functional.mse_loss(
                        features, target
                    )
                if self.lambda_so:
                    loss = (
                        loss
                        + self.lambda_so
                        * (
                            1 - torch.nn.functional.cosine_similarity(features, target)
                        ).mean()
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
        prototypes, counts = extract_prototypes(
            self.model,
            self.train_loader(task.client_id),
            self.num_classes,
            self.feature_dim,
            self.device,
        )
        if self.confidence_mode == "count":
            confidence = counts[:, None]
        elif self.confidence_mode == "log":
            confidence = torch.log1p(counts)[:, None]
        else:
            confidence = torch.ones_like(counts[:, None])
        return self.result(
            task,
            total / batches,
            {"sum": prototypes * confidence, "weight": confidence, "counts": counts},
        )


class Server(BaseServer):
    client_class = DFedSETClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        self.eta: float = args.eta
        self.trigger_mode: str = args.trigger_mode
        self.gamma_global: float = args.gamma_global
        self.relay = not args.no_relay
        self.redirect = not args.no_redirect
        self.weights = metropolis_hastings(adjacency(args)).to(self.device)
        self.sums = torch.zeros(self.num_clients, self.num_classes, args.feature_dim)
        self.confidences = torch.zeros(self.num_clients, self.num_classes, 1)
        self.counts = torch.zeros(self.num_clients, self.num_classes)
        self.consensus = torch.zeros_like(self.sums)
        self.local_gsd_ema = torch.full((self.num_clients,), 0.5)
        self.triggered = []

    def select_clients(self):
        """DFedSET 的原始协议每轮让全部节点更新 S/W/D 缓存。"""
        self.selected = list(range(self.num_clients))

    def training_payloads(self):
        return {i: {"consensus": self.consensus[i]} for i in self.selected}

    def apply_results(self, results):
        previous_consensus = self.consensus.clone()
        for i in self.selected:
            self.client_states[i] = results[i].state
            weight = results[i].payload["weight"]
            present = weight.squeeze(1) > 0
            self.sums[i, present] = results[i].payload["sum"][present]
            self.confidences[i, present] = weight[present]
            self.counts[i] = results[i].payload["counts"]
            if not self.relay:
                self.sums[i, ~present] = 0
                self.confidences[i, ~present] = 0
        local_prototypes = self.sums / self.confidences.clamp_min(1e-12)
        counts = self.counts
        probabilities = counts / counts.sum(1, keepdim=True).clamp_min(1e-12)
        valid = (local_prototypes.norm(dim=-1) > 1e-8) & (
            previous_consensus.norm(dim=-1) > 1e-8
        )
        similarity = torch.nn.functional.cosine_similarity(
            local_prototypes, previous_consensus, dim=-1
        )
        gsd = (probabilities * torch.where(valid, 1 - similarity, 0)).sum(1)
        gsd = torch.where(counts.sum(1) > 0, gsd, torch.ones_like(gsd))
        sums = self.weights.cpu() @ self.sums.reshape(self.num_clients, -1)
        confidences = self.weights.cpu() @ self.confidences.reshape(
            self.num_clients, -1
        )
        self.sums = sums.reshape_as(self.sums)
        self.confidences = confidences.reshape_as(self.confidences)
        self.consensus = self.sums / self.confidences.clamp_min(1e-12)
        round_index = len(self.accuracies)
        if self.trigger_mode == "all" or round_index < 2:
            active = torch.ones(self.num_clients, dtype=torch.bool)
            if round_index == 1:
                self.local_gsd_ema = gsd
        elif self.trigger_mode == "global":
            self.local_gsd_ema = self.eta * self.local_gsd_ema + (1 - self.eta) * gsd
            active = gsd > self.gamma_global
        else:
            self.local_gsd_ema = self.eta * self.local_gsd_ema + (1 - self.eta) * gsd
            active = gsd > self.local_gsd_ema
        if self.redirect:
            active = (self.weights.cpu().T @ active.float()) > 0
        self.triggered = torch.where(active)[0].tolist()
        self._mix_extractors(active)

    def _mix_extractors(self, active):
        states = self.client_states
        for name in states[0]:
            if not name.startswith("extractor.") or not torch.is_floating_point(
                states[0][name]
            ):
                continue
            values = torch.stack(
                [
                    states[i][name].to(self.device).flatten()
                    for i in range(self.num_clients)
                ]
            )
            matrix = self.weights.clone()
            if self.redirect:
                inactive = ~active.to(self.device)
                active_device = ~inactive
                matrix *= active_device[:, None] * active_device[None, :]
                redirected = (self.weights * inactive[None, :]).sum(1)
                matrix += torch.diag(redirected * active_device)
                matrix += torch.diag(inactive.float())
            else:
                matrix *= (
                    active.to(self.device)[:, None] * active.to(self.device)[None, :]
                )
                matrix = matrix / matrix.sum(1, keepdim=True).clamp_min(1e-12)
                matrix[~active.to(self.device)] = 0
                matrix[~active.to(self.device), ~active.to(self.device)] = 1
            mixed = matrix @ values
            for i in range(self.num_clients):
                states[i][name] = mixed[i].reshape_as(states[i][name]).cpu()

    def progress_fields(self):
        return {**super().progress_fields(), "triggered": len(self.triggered)}

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "weights": self.weights.cpu(),
            "sums": self.sums,
            "confidences": self.confidences,
            "counts": self.counts,
            "consensus": self.consensus,
            "local_gsd_ema": self.local_gsd_ema,
        }
