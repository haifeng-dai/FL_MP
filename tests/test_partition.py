"""仅验证标签和索引的联邦划分。"""

import pytest
import torch

from dataset.partition import (
    build_partition_plan,
    partition_with_plan,
    pathological_partition,
)


def test_pathological_partition_honors_classes_per_client() -> None:
    """每个客户端恰有指定数量类别，且每个样本只属于一个客户端。"""
    labels = torch.arange(4).repeat_interleave(20)
    indices = torch.arange(len(labels))
    parts = pathological_partition(
        indices,
        labels,
        num_clients=4,
        classes_per_client=2,
        seed=42,
    )

    observed = [set(labels[client_indices].tolist()) for client_indices in parts.values()]
    assert all(len(client_labels) == 2 for client_labels in observed)
    assert set.union(*observed) == {0, 1, 2, 3}
    assigned = torch.cat(list(parts.values()))
    assert len(assigned) == len(indices)
    assert len(torch.unique(assigned)) == len(indices)


def test_pathological_partition_rejects_insufficient_class_capacity() -> None:
    """客户端总类别容量不足时应明确失败。"""
    labels = torch.arange(10).repeat_interleave(2)
    with pytest.raises(ValueError, match="容量不足"):
        pathological_partition(
            torch.arange(len(labels)),
            labels,
            num_clients=2,
            classes_per_client=2,
            seed=42,
        )


def test_pathological_plan_keeps_train_and_test_client_classes_aligned() -> None:
    """同一病态计划在训练和测试上不得为客户端分配额外类别。"""
    labels = torch.arange(4).repeat_interleave(20)
    plan = build_partition_plan(
        labels, "pathological", num_clients=4, alpha=0.0, classes_per_client=2, seed=7
    )
    assert plan.client_classes is not None
    train = partition_with_plan(torch.arange(0, 48), labels[:48], plan, seed=7)
    test = partition_with_plan(torch.arange(48, 80), labels[48:], plan, seed=8)

    for client_id in train:
        allowed = plan.client_classes[client_id]
        assert set(labels[train[client_id]].tolist()).issubset(allowed)
        assert set(labels[test[client_id]].tolist()).issubset(allowed)


def test_dirichlet_plan_reuses_per_class_proportions() -> None:
    """训练和测试必须引用同一份 Dirichlet 类别比例计划。"""
    labels = torch.arange(3).repeat_interleave(30)
    plan = build_partition_plan(
        labels, "dirichlet", num_clients=3, alpha=0.5, classes_per_client=None, seed=3
    )

    assert plan.dirichlet_proportions is not None
    assert set(plan.dirichlet_proportions) == {0, 1, 2}
    train = partition_with_plan(torch.arange(45), labels[:45], plan, seed=3)
    test = partition_with_plan(torch.arange(45, 90), labels[45:], plan, seed=4)
    assert sum(len(indices) for indices in train.values()) == 45
    assert sum(len(indices) for indices in test.values()) == 45
