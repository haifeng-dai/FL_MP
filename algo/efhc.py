"""EF-HC：带事件触发通信的去中心化联邦学习。"""

import math

import torch

from .core import BaseClient, BaseServer, clone_state
from .core.decentralized import adjacency, metropolis_hastings, mix_states


def add_arguments(parser):
    parser.add_argument("--event-r", type=float, default=1.0)
    parser.add_argument("--bandwidth-mean", type=float, default=1.0)
    parser.add_argument("--bandwidth-std", type=float, default=0.0)


class EFHCClient(BaseClient):
    def train(self, task):
        loss = self.train_supervised(task.client_id)
        state = clone_state(self.model.state_dict())
        squared_norm = sum(
            (state[name] - task.payload["hat_state"][name]).float().square().sum()
            for name in state
        )
        size = sum(value.numel() for value in state.values())
        change = math.sqrt((squared_norm / size).item())
        return self.result(task, loss, {"change": change})


class Server(BaseServer):
    client_class = EFHCClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        self.event_r: float = args.event_r
        low: float = (1 - args.bandwidth_std) * args.bandwidth_mean
        high: float = (1 + args.bandwidth_std) * args.bandwidth_mean
        if low <= 0:
            raise ValueError("bandwidth-mean 与 bandwidth-std 必须生成正带宽")
        self.bandwidths = (low + (high - low) * torch.rand(self.num_clients)).tolist()
        self.rho = [1 / bandwidth for bandwidth in self.bandwidths]
        self.graph = adjacency(args)
        self.weights = metropolis_hastings(self.graph).to(self.device)
        self.hat_states = {
            client_id: clone_state(state)
            for client_id, state in self.client_states.items()
        }
        self.client_changes = {client_id: 0.0 for client_id in range(self.num_clients)}
        self.triggered_ids: list[int] = []

    def training_payloads(self):
        return {
            client_id: {"hat_state": self.hat_states[client_id]}
            for client_id in self.selected
        }

    def apply_results(self, results):
        self.update_client_states(results)
        for client_id in self.selected:
            self.client_changes[client_id] = results[client_id].payload["change"]
        gamma = 0.1 / math.sqrt(1 + len(self.accuracies))
        self.triggered_ids = [
            client_id
            for client_id in range(self.num_clients)
            if self.client_changes[client_id]
            >= self.event_r * self.rho[client_id] * gamma
        ]
        if not self.triggered_ids:
            return
        mixed = mix_states(
            [self.client_states[client_id] for client_id in range(self.num_clients)],
            self.weights,
            self.device,
        )
        for client_id in self.triggered_ids:
            self.client_states[client_id] = mixed[client_id]
            self.hat_states[client_id] = clone_state(mixed[client_id])

    def progress_fields(self):
        return {**super().progress_fields(), "triggered": len(self.triggered_ids)}

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "graph": self.graph.cpu(),
            "hat_states": self.hat_states,
            "client_changes": self.client_changes,
            "bandwidths": self.bandwidths,
        }
