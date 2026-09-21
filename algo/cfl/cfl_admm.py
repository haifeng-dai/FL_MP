"""CFL-ADMM：面向多去中心化边缘服务器的随机调度 ADMM。"""

import argparse
import copy

import torch
import torch.nn.functional as F

from ..core import BaseClient, BaseServer, clone_state
from ..core.decentralized import (
    add_topology_arguments,
    adjacency,
    validate_topology_arguments,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """添加 CFL-ADMM 的罚参数、本地求解参数和服务器拓扑。"""
    parser.add_argument("--num-servers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.3, help="用户独立激活概率")
    parser.add_argument("--sigma1", type=float, default=1.0, help="用户一致性罚参数")
    parser.add_argument("--sigma2", type=float, default=1.0, help="服务器一致性罚参数")
    parser.add_argument("--local-lr", type=float, default=1e-3)
    parser.add_argument("--local-max-steps", type=int, default=10)
    parser.add_argument("--local-tolerance", type=float, default=1e-3)
    add_topology_arguments(parser)


def laplacian_from_graph(graph: torch.Tensor) -> torch.Tensor:
    """从含自环的无向邻接矩阵构造论文中的无自环拉普拉斯矩阵。"""
    neighbors = graph.clone()
    neighbors.fill_diagonal_(0)
    return torch.diag(neighbors.sum(dim=1)) - neighbors


def proximal_diagonal(
    user_counts: torch.Tensor,
    degrees: torch.Tensor,
    alpha: float,
    sigma1: float,
    sigma2: float,
) -> torch.Tensor:
    """按论文式 (23) 计算服务器近端矩阵 D 的标量对角块。"""
    coefficient = (1 / alpha) * (1 / alpha**2 - 1) * (sigma1 / sigma2)
    return coefficient * user_counts + 1.5 * degrees


def server_primal_step(
    user_models: torch.Tensor,
    user_duals: torch.Tensor,
    server_models: torch.Tensor,
    graph_duals: torch.Tensor,
    assignments: torch.Tensor,
    user_counts: torch.Tensor,
    laplacian: torch.Tensor,
    diagonal: torch.Tensor,
    alpha: float,
    sigma1: float,
    sigma2: float,
) -> torch.Tensor:
    """以逐服务器闭式形式实现论文式 (14)。"""
    num_servers = len(server_models)
    model_sums = torch.zeros_like(server_models)
    dual_sums = torch.zeros_like(server_models)
    model_sums.index_add_(0, assignments, user_models)
    dual_sums.index_add_(0, assignments, user_duals)
    graph_mixing = laplacian @ server_models
    numerator = (
        alpha * sigma1 * model_sums
        + dual_sums
        - graph_duals
        + sigma2 * (diagonal[:, None] * server_models - graph_mixing)
    )
    denominator = alpha * sigma1 * user_counts + sigma2 * diagonal
    if len(denominator) != num_servers or torch.any(denominator <= 0):
        raise ValueError("服务器闭式更新的分母必须全部为正")
    return numerator / denominator[:, None]


def relaxed_user_dual_step(
    user_models: torch.Tensor,
    user_duals: torch.Tensor,
    server_models: torch.Tensor,
    assignments: torch.Tensor,
    alpha: float,
    sigma1: float,
) -> torch.Tensor:
    """实现论文式 (6) 的用户对偶过松弛更新。"""
    return user_duals + alpha * sigma1 * (
        user_models - server_models[assignments]
    )


def graph_dual_step(
    graph_duals: torch.Tensor,
    server_models: torch.Tensor,
    laplacian: torch.Tensor,
    sigma2: float,
) -> torch.Tensor:
    """等价更新 z=A^T beta，避免显式选择关联矩阵方向。"""
    return graph_duals + sigma2 * (laplacian @ server_models)


def _parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    parameters = [parameter.detach().reshape(-1) for parameter in model.parameters()]
    return torch.cat(parameters).cpu().clone() if parameters else torch.empty(0)


def _state_vector(
    state: dict[str, torch.Tensor],
    parameter_specs: tuple[tuple[str, torch.Size, int], ...],
) -> torch.Tensor:
    parameters = [state[name].reshape(-1) for name, _, _ in parameter_specs]
    return torch.cat(parameters).cpu().clone() if parameters else torch.empty(0)


def _state_from_vector(
    template: dict[str, torch.Tensor],
    parameter_specs: tuple[tuple[str, torch.Size, int], ...],
    vector: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state = clone_state(template)
    offset = 0
    for name, shape, size in parameter_specs:
        state[name] = vector[offset : offset + size].reshape(shape).clone()
        offset += size
    if offset != vector.numel():
        raise ValueError("参数向量维度与模型状态不匹配")
    return state


class CFLADMMClient(BaseClient):
    """在 CUDA 上以有限步全数据梯度下降近似求解用户 ADMM 子问题。"""

    def _objective_gradient(
        self,
        client_id: int,
        center: torch.Tensor,
        sigma1: float,
    ) -> tuple[float, float]:
        """计算经验风险与 ADMM 子问题的完整一阶残差。"""
        self.model.zero_grad(set_to_none=True)
        num_samples = len(self.train_sets[client_id])
        total_loss = 0.0
        for inputs, targets in self.train_loader(client_id):
            output = self.model(inputs.to(self.device, non_blocking=True))
            loss_sum = F.cross_entropy(
                output,
                targets.to(self.device, non_blocking=True),
                reduction="sum",
            )
            (loss_sum / num_samples).backward()
            total_loss += float(loss_sum.detach().item())

        offset = 0
        squared_residual = torch.zeros((), device=self.device)
        with torch.no_grad():
            for parameter in self.model.parameters():
                size = parameter.numel()
                center_parameter = center[offset : offset + size].reshape_as(parameter)
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                if self.weight_decay:
                    gradient = gradient + self.weight_decay * parameter
                gradient = gradient + sigma1 * (parameter - center_parameter)
                parameter.grad = gradient
                squared_residual.add_(gradient.float().square().sum())
                offset += size
        if offset != center.numel():
            raise ValueError("ADMM 近端中心的维度与模型不匹配")
        return total_loss / num_samples, float(torch.sqrt(squared_residual).item())

    def train(self, task):
        sigma1 = float(task.payload["sigma1"])
        local_lr = float(task.payload["local_lr"])
        max_steps = int(task.payload["local_max_steps"])
        tolerance = float(task.payload["local_tolerance"])
        server_vector = task.payload["server_vector"].to(self.device)
        dual_vector = task.payload["dual_vector"].to(self.device)
        center = server_vector - dual_vector / sigma1

        self.model.train()
        steps = 0
        for _ in range(max_steps):
            _, residual = self._objective_gradient(task.client_id, center, sigma1)
            if residual <= tolerance:
                break
            with torch.no_grad():
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-local_lr)
            steps += 1
        loss, residual = self._objective_gradient(task.client_id, center, sigma1)
        return self.result(
            task,
            loss,
            {"local_steps": steps, "residual": residual},
        )


class Server(BaseServer):
    """缓存所有用户变量并模拟多个仅与邻居通信的边缘服务器。"""

    client_class = CFLADMMClient

    def __init__(self, args):
        if args.num_servers < 1 or args.num_servers > args.num_clients:
            raise ValueError("num-servers 必须位于 [1, num-clients] 内")
        if not 0 < args.alpha <= 1:
            raise ValueError("alpha 必须位于 (0, 1] 内")
        if args.join_ratio != 1.0:
            raise ValueError("CFL-ADMM 使用 alpha 独立激活用户，请保持 --join-ratio 1")
        if args.sigma1 <= 0 or args.sigma2 <= 0:
            raise ValueError("sigma1 和 sigma2 必须为正数")
        if args.local_lr <= 0 or args.local_max_steps < 1:
            raise ValueError("local-lr 和 local-max-steps 必须为正数")
        if args.local_tolerance < 0:
            raise ValueError("local-tolerance 不得为负数")
        topology_args = copy.copy(args)
        topology_args.num_clients = args.num_servers
        validate_topology_arguments(topology_args)

        super().__init__(args)
        empty_clients = [
            client_id for client_id, dataset in self.train_sets.items() if not len(dataset)
        ]
        if empty_clients:
            self.pool.close()
            raise ValueError(f"CFL-ADMM 不支持空训练客户端：{empty_clients}")

        self.num_servers: int = args.num_servers
        self.alpha: float = args.alpha
        self.sigma1: float = args.sigma1
        self.sigma2: float = args.sigma2
        self.local_lr: float = args.local_lr
        self.local_max_steps: int = args.local_max_steps
        self.local_tolerance: float = args.local_tolerance
        self.client_servers = torch.arange(self.num_clients) % self.num_servers
        self.user_counts = torch.bincount(
            self.client_servers, minlength=self.num_servers
        ).float()

        self.graph = adjacency(topology_args)
        self.laplacian = laplacian_from_graph(self.graph)
        self.diagonal = proximal_diagonal(
            self.user_counts,
            self.laplacian.diag(),
            self.alpha,
            self.sigma1,
            self.sigma2,
        )

        self.state_template = clone_state(self.model.state_dict())
        self.parameter_specs = tuple(
            (name, parameter.shape, parameter.numel())
            for name, parameter in self.model.named_parameters()
        )
        initial_vector = _parameter_vector(self.model)
        self.user_models = initial_vector.repeat(self.num_clients, 1)
        self.user_duals = torch.zeros_like(self.user_models)
        self.server_models = initial_vector.repeat(self.num_servers, 1)
        self.graph_duals = torch.zeros_like(self.server_models)
        self.last_uploads = 0
        self.total_uploads = 0
        self.last_average_steps = 0.0
        self.last_average_residual = 0.0
        self.last_train_loss = 0.0

    def select_clients(self) -> None:
        """对每名用户执行独立 Bernoulli 激活，而不是固定数量抽样。"""
        self.selected = [
            client_id
            for client_id in range(self.num_clients)
            if self.random.random() < self.alpha
        ]

    def training_states(self):
        return {
            client_id: _state_from_vector(
                self.state_template,
                self.parameter_specs,
                self.user_models[client_id],
            )
            for client_id in self.selected
        }

    def training_payloads(self):
        return {
            client_id: {
                "server_vector": self.server_models[
                    self.client_servers[client_id]
                ].clone(),
                "dual_vector": self.user_duals[client_id].clone(),
                "sigma1": self.sigma1,
                "local_lr": self.local_lr,
                "local_max_steps": self.local_max_steps,
                "local_tolerance": self.local_tolerance,
            }
            for client_id in self.selected
        }

    def run_round(self):
        self.select_clients()
        results = self.run_clients() if self.selected else {}
        self.apply_results(results)
        if self.selected:
            self.last_train_loss = sum(
                result.loss for result in results.values()
            ) / len(results)
        return self.last_train_loss, self.evaluate()

    def apply_results(self, results):
        for client_id in self.selected:
            self.user_models[client_id] = _state_vector(
                results[client_id].state, self.parameter_specs
            )

        device = self.device
        assignments = self.client_servers.to(device)
        new_server_models = server_primal_step(
            self.user_models.to(device),
            self.user_duals.to(device),
            self.server_models.to(device),
            self.graph_duals.to(device),
            assignments,
            self.user_counts.to(device),
            self.laplacian.to(device),
            self.diagonal.to(device),
            self.alpha,
            self.sigma1,
            self.sigma2,
        )
        new_graph_duals = graph_dual_step(
            self.graph_duals.to(device),
            new_server_models,
            self.laplacian.to(device),
            self.sigma2,
        )
        new_user_duals = relaxed_user_dual_step(
            self.user_models.to(device),
            self.user_duals.to(device),
            new_server_models,
            assignments,
            self.alpha,
            self.sigma1,
        )
        self.server_models = new_server_models.cpu()
        self.graph_duals = new_graph_duals.cpu()
        self.user_duals = new_user_duals.cpu()
        self.model.load_state_dict(
            _state_from_vector(
                self.state_template,
                self.parameter_specs,
                self.server_models.mean(dim=0),
            )
        )

        self.last_uploads = len(self.selected)
        self.total_uploads += self.last_uploads
        if self.selected:
            self.last_average_steps = sum(
                results[client_id].payload["local_steps"]
                for client_id in self.selected
            ) / len(self.selected)
            self.last_average_residual = sum(
                results[client_id].payload["residual"]
                for client_id in self.selected
            ) / len(self.selected)
        else:
            self.last_average_steps = 0.0
            self.last_average_residual = 0.0

    def progress_fields(self):
        return {
            "uploads": self.last_uploads,
            "upload_rate": f"{self.last_uploads / self.num_clients:.3f}",
            "local_steps": f"{self.last_average_steps:.2f}",
            "residual": f"{self.last_average_residual:.3e}",
            "total_uploads": self.total_uploads,
        }

    def checkpoint_state(self):
        return {
            "graph": self.graph.cpu(),
            "laplacian": self.laplacian.cpu(),
            "diagonal": self.diagonal.cpu(),
            "client_servers": self.client_servers.cpu(),
            "user_models": self.user_models.cpu(),
            "user_duals": self.user_duals.cpu(),
            "server_models": self.server_models.cpu(),
            "graph_duals": self.graph_duals.cpu(),
            "total_uploads": self.total_uploads,
        }
