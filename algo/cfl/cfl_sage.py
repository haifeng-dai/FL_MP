"""论文 CFL-SAGA：事件触发 SAGA 与多服务器梯度跟踪。"""

import argparse
import copy
import math
from dataclasses import dataclass

import torch
from torch.utils.data._utils.collate import default_collate

from ..core import BaseClient, BaseServer, clone_state
from ..core.decentralized import (
    add_topology_arguments,
    adjacency,
    metropolis_hastings,
    validate_topology_arguments,
)
from ..core.protocol import ClientResult


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """添加论文中的服务器数量、步长、触发参数及服务器拓扑。"""
    parser.add_argument("--num-servers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--rho", type=float, default=10.0)
    add_topology_arguments(parser)


def saga_estimate(
    new_gradient: torch.Tensor,
    old_gradient: torch.Tensor,
    gradient_sum: torch.Tensor,
    num_batches: int,
) -> torch.Tensor:
    """按论文式 (16) 计算一个用户的方差缩减随机梯度。"""
    return num_batches * (new_gradient - old_gradient) + gradient_sum


def event_triggered(
    innovation: torch.Tensor, threshold: float, rho: float
) -> bool:
    """实现论文式 (17) 的严格大于触发条件。"""
    return bool(innovation.float().square().sum().item() > rho * threshold)


def gradient_tracking_step(
    trackers: torch.Tensor,
    old_gradients: torch.Tensor,
    new_gradients: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """实现论文算法 3 第 5 步，便于独立验证更新方向。"""
    return weights @ trackers + new_gradients - old_gradients


def _parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    """将模型参数展平为独立 CPU Tensor，不包含非优化缓冲区。"""
    parameters = [parameter.detach().reshape(-1) for parameter in model.parameters()]
    if not parameters:
        return torch.empty(0)
    return torch.cat(parameters).cpu().clone()


def _gradient_vector(model: torch.nn.Module) -> torch.Tensor:
    """按参数顺序提取当前梯度为独立 CPU Tensor。"""
    gradients = []
    for parameter in model.parameters():
        if parameter.grad is None:
            gradients.append(torch.zeros_like(parameter).reshape(-1))
        else:
            gradients.append(parameter.grad.detach().reshape(-1))
    if not gradients:
        return torch.empty(0)
    return torch.cat(gradients).cpu().clone()


@torch.no_grad()
def _load_parameter_vector(model: torch.nn.Module, vector: torch.Tensor) -> None:
    """把展平向量复制回模型参数，并检查维度完全匹配。"""
    offset = 0
    for parameter in model.parameters():
        size = parameter.numel()
        parameter.copy_(
            vector[offset : offset + size].reshape_as(parameter).to(parameter.device)
        )
        offset += size
    if offset != vector.numel():
        raise ValueError("参数向量维度与模型不匹配")


def _state_from_vector(
    template: dict[str, torch.Tensor],
    parameter_specs: tuple[tuple[str, torch.Size, int], ...],
    vector: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """用参数向量重建可跨进程发送的完整模型状态。"""
    state = clone_state(template)
    offset = 0
    for name, shape, size in parameter_specs:
        state[name] = vector[offset : offset + size].reshape(shape).clone()
        offset += size
    if offset != vector.numel():
        raise ValueError("参数向量维度与状态模板不匹配")
    return state


@dataclass
class _SAGAMemory:
    """固定驻留在用户所属 Worker 中的 SAGA 与通信缓存。"""

    gradient_table: list[torch.Tensor]
    gradient_sum: torch.Tensor
    committed_gradient: torch.Tensor
    generator: torch.Generator


class CFLSAGAClient(BaseClient):
    """每轮计算一个 SAGA 候选梯度，并按 CTUS 决定是否上传创新量。"""

    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.memories: dict[int, _SAGAMemory] = {}

    def _batch(self, client_id: int, batch_index: int):
        """按稳定顺序读取指定 mini-batch，使 SAGA 表索引跨轮不变。"""
        dataset = self.train_sets[client_id]
        start = batch_index * self.batch_size
        stop = min(start + self.batch_size, len(dataset))
        return default_collate([dataset[index] for index in range(start, stop)])

    def _batch_gradient(self, inputs, targets) -> tuple[torch.Tensor, float]:
        """计算一个固定 mini-batch 损失及其完整参数梯度。"""
        self.model.zero_grad(set_to_none=True)
        output = self.model(inputs.to(self.device, non_blocking=True))
        loss = self.ce_loss(output, targets.to(self.device, non_blocking=True))
        if self.weight_decay:
            penalty = sum(
                parameter.float().square().sum()
                for parameter in self.model.parameters()
            )
            loss = loss + 0.5 * self.weight_decay * penalty
        loss.backward()
        return _gradient_vector(self.model), float(loss.detach().item())

    def _initialize_memory(self, task) -> _SAGAMemory:
        """在论文规定的 phi=0 处初始化全部 mini-batch 梯度表。"""
        dataset_size = len(self.train_sets[task.client_id])
        if dataset_size == 0:
            raise ValueError(f"客户端 {task.client_id} 没有训练样本")
        num_batches = math.ceil(dataset_size / self.batch_size)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
        table: list[torch.Tensor] = []
        gradient_sum: torch.Tensor | None = None
        for batch_index in range(num_batches):
            gradient = self._batch_gradient(
                *self._batch(task.client_id, batch_index)
            )[0]
            table.append(gradient)
            if gradient_sum is None:
                gradient_sum = torch.zeros_like(gradient)
            gradient_sum.add_(gradient)
        if gradient_sum is None:
            raise RuntimeError("非空客户端未生成任何 mini-batch")
        self.model.load_state_dict(task.state)
        self.model.to(self.device)
        generator = torch.Generator().manual_seed(task.seed + task.client_id)
        return _SAGAMemory(
            gradient_table=table,
            gradient_sum=gradient_sum,
            committed_gradient=torch.zeros_like(gradient_sum),
            generator=generator,
        )

    def train(self, task):
        memory = self.memories.get(task.client_id)
        if memory is None:
            memory = self._initialize_memory(task)
            self.memories[task.client_id] = memory

        num_batches = len(memory.gradient_table)
        batch_index = int(
            torch.randint(num_batches, (1,), generator=memory.generator).item()
        )
        new_gradient, loss = self._batch_gradient(
            *self._batch(task.client_id, batch_index)
        )
        old_gradient = memory.gradient_table[batch_index]
        candidate = saga_estimate(
            new_gradient, old_gradient, memory.gradient_sum, num_batches
        )
        memory.gradient_sum.add_(new_gradient - old_gradient)
        memory.gradient_table[batch_index] = new_gradient

        innovation = candidate - memory.committed_gradient
        triggered = event_triggered(
            innovation,
            float(task.payload["threshold"]),
            float(task.payload["rho"]),
        )
        delta = None
        if triggered:
            delta = innovation.detach().cpu().clone()
            memory.committed_gradient = candidate.detach().cpu().clone()

        return ClientResult(
            client_id=task.client_id,
            state={},
            num_samples=len(self.train_sets[task.client_id]),
            loss=loss,
            payload={"triggered": triggered, "delta": delta},
        )


class Server(BaseServer):
    """在单主进程中精确模拟论文的多个互联边缘服务器。"""

    client_class = CFLSAGAClient

    def __init__(self, args):
        if args.num_servers < 1 or args.num_servers > args.num_clients:
            raise ValueError("num-servers 必须位于 [1, num-clients] 内")
        if args.alpha <= 0 or args.rho < 0:
            raise ValueError("alpha 必须为正数，rho 不得为负数")
        if args.join_ratio != 1.0:
            raise ValueError("CFL-SAGA 按论文要求每轮必须调度全部用户")
        topology_args = copy.copy(args)
        topology_args.num_clients = args.num_servers
        validate_topology_arguments(topology_args)

        super().__init__(args)
        empty_clients = [
            client_id for client_id, dataset in self.train_sets.items() if not len(dataset)
        ]
        if empty_clients:
            self.pool.close()
            raise ValueError(f"CFL-SAGA 不支持空训练客户端：{empty_clients}")

        self.num_servers: int = args.num_servers
        self.alpha: float = args.alpha
        self.rho: float = args.rho
        self.client_servers = {
            client_id: client_id % self.num_servers
            for client_id in range(self.num_clients)
        }
        self.graph = adjacency(topology_args)
        self.weights = metropolis_hastings(self.graph)

        initial_state = clone_state(self.model.state_dict())
        named_parameters = dict(self.model.named_parameters())
        self.parameter_specs = tuple(
            (name, parameter.shape, parameter.numel())
            for name, parameter in named_parameters.items()
        )
        self.state_template = initial_state
        initial_vector = _parameter_vector(self.model)
        self.server_models = initial_vector.repeat(self.num_servers, 1)
        self.trackers = torch.zeros_like(self.server_models)
        self.server_gradients = torch.zeros_like(self.server_models)
        self._next_server_models = self.server_models.clone()
        self._thresholds = torch.zeros(self.num_servers)
        self._training_states: dict[int, dict[str, torch.Tensor]] = {}
        self.total_uploads = 0
        self.last_uploads = 0

    def select_clients(self) -> None:
        """论文算法 3 要求所有用户每轮都计算候选 SAGA 梯度。"""
        self.selected = list(range(self.num_clients))

    def _prepare_round(self) -> None:
        """执行论文算法 3 的模型更新及事件阈值计算。"""
        weights = self.weights.to(self.device)
        models = self.server_models.to(self.device)
        trackers = self.trackers.to(self.device)
        next_models = weights @ models - self.alpha * trackers
        discrepancies = weights @ next_models - next_models
        self._next_server_models = next_models.cpu()
        self._thresholds = discrepancies.float().square().sum(dim=1).cpu()
        server_states = {
            server_id: _state_from_vector(
                self.state_template,
                self.parameter_specs,
                self._next_server_models[server_id],
            )
            for server_id in range(self.num_servers)
        }
        self._training_states = {
            client_id: server_states[self.client_servers[client_id]]
            for client_id in range(self.num_clients)
        }

    def run_round(self):
        self._prepare_round()
        return super().run_round()

    def training_states(self):
        return self._training_states

    def training_payloads(self):
        return {
            client_id: {
                "threshold": float(
                    self._thresholds[self.client_servers[client_id]].item()
                ),
                "rho": self.rho,
            }
            for client_id in self.selected
        }

    def run_clients(self):
        """固定用户所属 Worker，以保存用户侧 SAGA 梯度表。"""
        return self.pool.run_affined(self.build_tasks())

    def apply_results(self, results):
        """聚合触发创新量，并完成服务器梯度跟踪更新。"""
        gradient_deltas = torch.zeros_like(self.server_gradients)
        uploads = 0
        for client_id in self.selected:
            payload = results[client_id].payload
            if not payload["triggered"]:
                continue
            delta = payload["delta"]
            if not isinstance(delta, torch.Tensor) or delta.device.type != "cpu":
                raise ValueError("用户上传的梯度创新量必须是 CPU Tensor")
            gradient_deltas[self.client_servers[client_id]].add_(delta)
            uploads += 1

        old_gradients = self.server_gradients
        new_gradients = old_gradients + gradient_deltas
        self.trackers = gradient_tracking_step(
            self.trackers.to(self.device),
            old_gradients.to(self.device),
            new_gradients.to(self.device),
            self.weights.to(self.device),
        ).cpu()
        self.server_gradients = new_gradients
        self.server_models = self._next_server_models
        average_model = self.server_models.mean(dim=0)
        self.model.load_state_dict(
            _state_from_vector(
                self.state_template, self.parameter_specs, average_model
            )
        )
        self.last_uploads = uploads
        self.total_uploads += uploads

    def progress_fields(self):
        return {
            "uploads": self.last_uploads,
            "upload_rate": f"{self.last_uploads / self.num_clients:.3f}",
            "total_uploads": self.total_uploads,
        }

    def checkpoint_state(self):
        return {
            "graph": self.graph.cpu(),
            "weights": self.weights.cpu(),
            "server_models": self.server_models.cpu(),
            "trackers": self.trackers.cpu(),
            "server_gradients": self.server_gradients.cpu(),
            "client_servers": self.client_servers,
            "total_uploads": self.total_uploads,
        }
