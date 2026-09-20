"""去中心化算法的拓扑和客户端状态混合工具。"""

import argparse

import torch

from .state import clone_state


def add_topology_arguments(parser: argparse.ArgumentParser) -> None:
    """添加去中心化算法共享的通信拓扑参数。"""
    parser.add_argument(
        "--adj-type",
        choices=("ring", "complete", "random", "small_world", "scale_free", "star"),
        default="ring",
    )
    parser.add_argument("--edge-p", type=float, default=0.2)
    parser.add_argument("--k-small-world", type=int, default=4)
    parser.add_argument("--m-scale-free", type=int, default=2)


def validate_topology_arguments(args: argparse.Namespace) -> None:
    """验证去中心化通信拓扑参数。"""
    if not 0 < args.edge_p <= 1:
        raise ValueError("edge-p 必须在 (0, 1] 内")


def adjacency(args) -> torch.Tensor:
    """按配置生成带自环的无向邻接矩阵。"""
    count = args.num_clients
    matrix = torch.eye(count)
    if count == 1 or args.adj_type == "complete":
        return torch.ones(count, count)
    if args.adj_type == "ring":
        indices = torch.arange(count)
        matrix[indices, (indices + 1) % count] = 1
        matrix[indices, (indices - 1) % count] = 1
    elif args.adj_type == "star":
        matrix[0, :] = matrix[:, 0] = 1
    elif args.adj_type == "random":
        for _ in range(100):
            upper = torch.rand(count, count) < args.edge_p
            matrix = torch.triu(upper, diagonal=1).float()
            matrix = matrix + matrix.T + torch.eye(count)
            if _is_connected(matrix):
                break
    elif args.adj_type == "small_world":
        half = min(args.k_small_world // 2, (count - 1) // 2)
        for distance in range(1, half + 1):
            indices = torch.arange(count)
            matrix[indices, (indices + distance) % count] = 1
            matrix[indices, (indices - distance) % count] = 1
        for node in range(count):
            for distance in range(1, half + 1):
                neighbor = (node + distance) % count
                if torch.rand(()) < args.edge_p:
                    matrix[node, neighbor] = matrix[neighbor, node] = 0
                    candidates = torch.where(matrix[node] == 0)[0]
                    candidates = candidates[candidates != node]
                    if len(candidates):
                        replacement = candidates[torch.randint(len(candidates), ())]
                        matrix[node, replacement] = matrix[replacement, node] = 1
    elif args.adj_type == "scale_free":
        initial = min(args.m_scale_free + 1, count)
        matrix[:initial, :initial] = 1
        degree = matrix.sum(dim=1)
        for node in range(initial, count):
            choices = min(args.m_scale_free, node)
            parents = torch.multinomial(degree[:node], choices, replacement=False)
            matrix[node, parents] = matrix[parents, node] = 1
            degree = matrix.sum(dim=1)
    else:
        raise ValueError(f"不支持的拓扑类型：{args.adj_type}")
    return matrix


def _is_connected(graph: torch.Tensor) -> bool:
    """用张量邻接矩阵检查无向图连通性，避免引入 networkx。"""
    reached = torch.zeros(len(graph), dtype=torch.bool)
    reached[0] = True
    while True:
        expanded = (graph[reached].sum(dim=0) > 0) | reached
        if torch.equal(expanded, reached):
            return bool(reached.all())
        reached = expanded


def metropolis_hastings(graph: torch.Tensor) -> torch.Tensor:
    """从邻接矩阵构造逐行随机 MH 混合矩阵。"""
    degree = graph.sum(dim=1)
    weights = graph / torch.maximum(degree[:, None], degree[None, :])
    weights.fill_diagonal_(0)
    return weights + torch.diag(1 - weights.sum(dim=1))


def sinkhorn(graph: torch.Tensor, iterations: int = 100) -> torch.Tensor:
    """将图的非负权重近似归一化为双随机矩阵。"""
    weights = graph.float().clone()
    for _ in range(iterations):
        weights /= weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights /= weights.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return weights


def mix_states(states, weights: torch.Tensor, device: torch.device):
    """按混合矩阵为每个客户端生成不同的独立 CPU 模型状态。"""
    result = [clone_state(state) for state in states]
    for name in result[0]:
        values = [state[name].to(device) for state in states]
        if not torch.is_floating_point(values[0]):
            continue
        flat = torch.stack([value.reshape(-1) for value in values])
        mixed = weights.to(device) @ flat
        for client_id, value in enumerate(values):
            result[client_id][name] = mixed[client_id].reshape(value.shape).cpu()
    return result
