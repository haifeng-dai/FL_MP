"""仅基于标签与索引的联邦数据划分。"""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class PartitionPlan:
    """训练与测试共用的客户端标签分配计划。"""

    method: str
    num_clients: int
    dirichlet_proportions: dict[int, torch.Tensor] | None = None
    client_classes: tuple[frozenset[int], ...] | None = None


def split_train_test_indices(
    labels: torch.Tensor, test_ratio: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机打乱样本索引后切分训练与测试索引。"""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(labels), generator=generator)
    test_size = int(len(indices) * test_ratio)
    return indices[test_size:], indices[:test_size]


def partition_indices(
    indices: torch.Tensor,
    labels: torch.Tensor,
    method: str,
    num_clients: int,
    alpha: float,
    classes_per_client: int | None,
    seed: int,
) -> dict[int, torch.Tensor]:
    """只按 ``labels`` 与 ``indices`` 产生客户端样本索引。"""
    plan = build_partition_plan(
        labels, method, num_clients, alpha, classes_per_client, seed
    )
    return partition_with_plan(indices, labels, plan, seed)


def build_partition_plan(
    labels: torch.Tensor,
    method: str,
    num_clients: int,
    alpha: float,
    classes_per_client: int | None,
    seed: int,
) -> PartitionPlan:
    """仅根据标签生成可同时应用于训练和测试的分配计划。"""
    classes = torch.unique(labels, sorted=True)
    if method == "iid":
        return PartitionPlan(method, num_clients)
    if method == "dirichlet":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            proportions = {
                int(label): torch.distributions.Dirichlet(
                    torch.full((num_clients,), alpha)
                ).sample()
                for label in classes
            }
        return PartitionPlan(method, num_clients, dirichlet_proportions=proportions)
    if method == "pathological":
        if classes_per_client is None:
            raise ValueError("病态划分必须指定 --classes-per-client")
        if classes_per_client > len(classes):
            raise ValueError("classes-per-client 不能大于数据集类别数")
        if num_clients * classes_per_client < len(classes):
            raise ValueError("客户端类别容量不足，无法覆盖数据集全部类别")
        generator = torch.Generator().manual_seed(seed)
        assigned = _assign_client_classes(
            classes, num_clients, classes_per_client, generator
        )
        return PartitionPlan(
            method,
            num_clients,
            client_classes=tuple(frozenset(values) for values in assigned),
        )
    raise ValueError(f"不支持的划分策略：{method}")


def partition_with_plan(
    indices: torch.Tensor,
    labels: torch.Tensor,
    plan: PartitionPlan,
    seed: int,
) -> dict[int, torch.Tensor]:
    """将计划应用到一组索引；训练和测试均调用此函数。"""
    if plan.method == "iid":
        return iid_partition(indices, labels, plan.num_clients, seed)
    if plan.method == "dirichlet":
        return dirichlet_partition_with_plan(
            indices, labels, plan.num_clients, plan.dirichlet_proportions, seed
        )
    return pathological_partition_with_plan(
        indices, labels, plan.num_clients, plan.client_classes, seed
    )


def assemble_subsets(
    dataset: Dataset,
    client_indices: dict[int, torch.Tensor],
) -> dict[int, Subset]:
    """将客户端索引组装成 ``Subset``；不参与任何划分决策。"""
    return {
        client_id: Subset(dataset, indices.tolist())
        for client_id, indices in client_indices.items()
    }


def iid_partition(
    indices: torch.Tensor, labels: torch.Tensor, num_clients: int, seed: int
) -> dict[int, torch.Tensor]:
    """在每个类别内部均分样本。"""
    generator = torch.Generator().manual_seed(seed)
    parts: list[list[torch.Tensor]] = [[] for _ in range(num_clients)]
    for label in torch.unique(labels, sorted=True):
        class_indices = indices[labels == label]
        shuffled = class_indices[
            torch.randperm(len(class_indices), generator=generator)
        ]
        for client_id, chunk in enumerate(torch.tensor_split(shuffled, num_clients)):
            parts[client_id].append(chunk)
    return _merge_parts(parts)


def dirichlet_partition(
    indices: torch.Tensor,
    labels: torch.Tensor,
    num_clients: int,
    alpha: float,
    seed: int,
) -> dict[int, torch.Tensor]:
    """按类别独立抽取 Dirichlet 比例；空客户端按设计保留。"""
    plan = build_partition_plan(labels, "dirichlet", num_clients, alpha, None, seed)
    return partition_with_plan(indices, labels, plan, seed)


def dirichlet_partition_with_plan(
    indices: torch.Tensor,
    labels: torch.Tensor,
    num_clients: int,
    proportions_by_class: dict[int, torch.Tensor] | None,
    seed: int,
) -> dict[int, torch.Tensor]:
    """用固定的每类 Dirichlet 比例划分一组索引。"""
    if proportions_by_class is None:
        raise ValueError("Dirichlet 划分计划缺少类别比例")
    generator = torch.Generator().manual_seed(seed)
    parts: list[list[torch.Tensor]] = [[] for _ in range(num_clients)]
    for label in torch.unique(labels, sorted=True):
        class_indices = indices[labels == label]
        shuffled = class_indices[torch.randperm(len(class_indices), generator=generator)]
        proportions = proportions_by_class[int(label)]
        boundaries = (torch.cumsum(proportions, dim=0)[:-1] * len(shuffled)).to(
            torch.int64
        )
        for client_id, chunk in enumerate(torch.tensor_split(shuffled, boundaries.tolist())):
            parts[client_id].append(chunk)
    return _merge_parts(parts)


def pathological_partition(
    indices: torch.Tensor,
    labels: torch.Tensor,
    num_clients: int,
    classes_per_client: int,
    seed: int,
) -> dict[int, torch.Tensor]:
    """让每个客户端拥有指定数量类别，并在类别内近似均分样本。"""
    plan = build_partition_plan(
        labels, "pathological", num_clients, 0.0, classes_per_client, seed
    )
    return partition_with_plan(indices, labels, plan, seed)


def pathological_partition_with_plan(
    indices: torch.Tensor,
    labels: torch.Tensor,
    num_clients: int,
    client_classes: tuple[frozenset[int], ...] | None,
    seed: int,
) -> dict[int, torch.Tensor]:
    """使用固定客户端类别集合划分一组索引。"""
    if client_classes is None:
        raise ValueError("病态划分计划缺少客户端类别集合")
    generator = torch.Generator().manual_seed(seed)
    classes = torch.unique(labels, sorted=True)
    parts: list[list[torch.Tensor]] = [[] for _ in range(num_clients)]
    for label in classes:
        class_indices = indices[labels == label]
        shuffled = class_indices[
            torch.randperm(len(class_indices), generator=generator)
        ]
        assigned_clients = [
            client_id
            for client_id, assigned in enumerate(client_classes)
            if int(label) in assigned
        ]
        for client_id, chunk in zip(
            assigned_clients,
            torch.tensor_split(shuffled, len(assigned_clients)),
            strict=True,
        ):
            parts[client_id].append(chunk)
    return _merge_parts(parts)


def _assign_client_classes(
    classes: torch.Tensor,
    num_clients: int,
    classes_per_client: int,
    generator: torch.Generator,
) -> list[set[int]]:
    """先覆盖全部类别，再填满每个客户端的类别配额。"""
    assignments: list[set[int]] = [set() for _ in range(num_clients)]
    shuffled_classes = classes[torch.randperm(len(classes), generator=generator)]
    for label in shuffled_classes.tolist():
        candidates = [
            client_id
            for client_id, assigned in enumerate(assignments)
            if len(assigned) < classes_per_client
        ]
        minimum = min(len(assignments[client_id]) for client_id in candidates)
        balanced = [
            client_id
            for client_id in candidates
            if len(assignments[client_id]) == minimum
        ]
        choice = torch.randint(len(balanced), (1,), generator=generator).item()
        assignments[balanced[choice]].add(label)
    for assigned in assignments:
        candidates = [int(label) for label in classes.tolist() if label not in assigned]
        permutation = torch.randperm(len(candidates), generator=generator).tolist()
        assigned.update(
            candidates[index]
            for index in permutation[: classes_per_client - len(assigned)]
        )
    return assignments


def _merge_parts(parts: list[list[torch.Tensor]]) -> dict[int, torch.Tensor]:
    """合并每个客户端从各类别取得的索引块。"""
    return {
        client_id: torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.int64)
        for client_id, chunks in enumerate(parts)
    }
