"""基础 CFL-FedAvg：服务器内 FedAvg，服务器间同步模型平均。"""

import argparse
import copy

import torch

from ..core import BaseClient, BaseServer, clone_state
from ..core.decentralized import (
    add_topology_arguments,
    adjacency,
    metropolis_hastings,
    mix_states,
    validate_topology_arguments,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """添加服务器数量、每轮混合次数和服务器拓扑参数。"""
    parser.add_argument("--num-servers", type=int, default=4)
    parser.add_argument(
        "--server-mix-steps",
        type=int,
        default=1,
        help="每轮本地 FedAvg 后的服务器同步模型平均次数",
    )
    add_topology_arguments(parser)


def average_states(
    states: list[dict[str, torch.Tensor]], weights: list[float]
) -> dict[str, torch.Tensor]:
    """按给定权重平均模型；非浮点缓冲区沿用第一份状态。"""
    if not states or len(states) != len(weights):
        raise ValueError("模型状态和聚合权重必须非空且数量一致")
    if any(weight < 0 for weight in weights):
        raise ValueError("聚合权重不得为负数")
    total = sum(weights)
    if total <= 0:
        raise ValueError("聚合权重之和必须为正数")
    normalized = [weight / total for weight in weights]
    result = clone_state(states[0])
    for name, tensor in result.items():
        if not torch.is_floating_point(tensor):
            continue
        tensor.zero_()
        for state, weight in zip(states, normalized, strict=True):
            tensor.add_(state[name], alpha=weight)
    return result


def mix_server_models(
    states: list[dict[str, torch.Tensor]],
    weights: torch.Tensor,
    steps: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """用同步旧状态连续执行指定次数的一跳服务器模型平均。"""
    if steps < 0:
        raise ValueError("服务器混合次数不得为负数")
    mixed = [clone_state(state) for state in states]
    for _ in range(steps):
        mixed = mix_states(mixed, weights, device)
    return mixed


class CFLFedAvgClient(BaseClient):
    """执行标准本地 SGD，并上传完整模型。"""

    def train(self, task):
        loss = self.train_supervised(task.client_id)
        return self.result(task, loss)


class Server(BaseServer):
    """维护多份服务器模型并执行分层 FedAvg。"""

    client_class = CFLFedAvgClient

    def __init__(self, args):
        if args.num_servers < 1 or args.num_servers > args.num_clients:
            raise ValueError("num-servers 必须位于 [1, num-clients] 内")
        if args.server_mix_steps < 0:
            raise ValueError("server-mix-steps 不得为负数")
        topology_args = copy.copy(args)
        topology_args.num_clients = args.num_servers
        validate_topology_arguments(topology_args)

        super().__init__(args)
        empty_clients = [
            client_id for client_id, dataset in self.train_sets.items() if not len(dataset)
        ]
        if empty_clients:
            self.pool.close()
            raise ValueError(f"CFL-FedAvg 不支持空训练客户端：{empty_clients}")

        self.num_servers: int = args.num_servers
        self.server_mix_steps: int = args.server_mix_steps
        self.client_servers = {
            client_id: client_id % self.num_servers
            for client_id in range(self.num_clients)
        }
        self.server_clients = {
            server_id: [
                client_id
                for client_id, assigned in self.client_servers.items()
                if assigned == server_id
            ]
            for server_id in range(self.num_servers)
        }
        self.graph = adjacency(topology_args)
        self.weights = metropolis_hastings(self.graph)
        initial_state = clone_state(self.model.state_dict())
        self.server_states = [
            clone_state(initial_state) for _ in range(self.num_servers)
        ]
        self.selected_by_server: dict[int, list[int]] = {}
        self.last_user_uploads = 0
        self.total_user_uploads = 0
        directed_links = self.graph.clone()
        directed_links.fill_diagonal_(0)
        self.directed_server_links = int(directed_links.sum().item())
        self.last_server_transmissions = 0
        self.total_server_transmissions = 0

    def select_clients(self) -> None:
        """在每个服务器内部独立抽样，并保证每个服务器至少一名用户。"""
        self.selected_by_server = {}
        for server_id, clients in self.server_clients.items():
            count = max(1, int(len(clients) * self.join_ratio))
            self.selected_by_server[server_id] = sorted(
                self.random.sample(clients, count)
            )
        self.selected = sorted(
            client_id
            for clients in self.selected_by_server.values()
            for client_id in clients
        )

    def training_states(self):
        return {
            client_id: self.server_states[self.client_servers[client_id]]
            for client_id in self.selected
        }

    def apply_results(self, results):
        locally_aggregated: list[dict[str, torch.Tensor]] = []
        for server_id in range(self.num_servers):
            clients = self.selected_by_server[server_id]
            locally_aggregated.append(
                average_states(
                    [results[client_id].state for client_id in clients],
                    [float(results[client_id].num_samples) for client_id in clients],
                )
            )
        self.server_states = mix_server_models(
            locally_aggregated,
            self.weights,
            self.server_mix_steps,
            self.device,
        )
        self.model.load_state_dict(
            average_states(
                self.server_states,
                [1.0] * self.num_servers,
            )
        )

        self.last_user_uploads = len(self.selected)
        self.total_user_uploads += self.last_user_uploads
        self.last_server_transmissions = (
            self.server_mix_steps * self.directed_server_links
        )
        self.total_server_transmissions += self.last_server_transmissions

    def progress_fields(self):
        return {
            "user_uploads": self.last_user_uploads,
            "mix_steps": self.server_mix_steps,
            "server_tx": self.last_server_transmissions,
            "total_server_tx": self.total_server_transmissions,
        }

    def checkpoint_state(self):
        return {
            "graph": self.graph.cpu(),
            "weights": self.weights.cpu(),
            "client_servers": self.client_servers,
            "server_states": [clone_state(state) for state in self.server_states],
            "total_user_uploads": self.total_user_uploads,
            "total_server_transmissions": self.total_server_transmissions,
        }
