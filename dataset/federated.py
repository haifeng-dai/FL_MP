"""数据集下载/加载与纯索引划分的组装入口。"""

from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .partition import (
    assemble_subsets,
    build_partition_plan,
    partition_with_plan,
    split_train_test_indices,
)

DATASETS = {
    "cifar10": (datasets.CIFAR10, 10),
    "cifar100": (datasets.CIFAR100, 100),
    "mnist": (datasets.MNIST, 10),
}


def load_federated_data(
    name: str,
    data_root: str,
    test_ratio: float,
    partition: str,
    num_clients: int,
    dirichlet: float,
    classes_per_client: int | None,
    seed: int,
) -> tuple[dict[int, Subset], dict[int, Subset], int]:
    """下载数据，并按同一客户端计划组装训练与测试 ``Subset``。"""
    dataset_cls, num_classes = DATASETS[name]
    root = Path(data_root) / name
    transform = transforms.ToTensor()
    train = dataset_cls(root=root, train=True, download=True, transform=transform)
    test = dataset_cls(root=root, train=False, download=True, transform=transform)
    all_data = ConcatDataset((train, test))
    labels = torch.cat(
        (
            torch.as_tensor(train.targets, dtype=torch.int64),
            torch.as_tensor(test.targets, dtype=torch.int64),
        )
    )
    train_indices, test_indices = split_train_test_indices(labels, test_ratio, seed)
    plan = build_partition_plan(
        labels,
        partition,
        num_clients,
        dirichlet,
        classes_per_client,
        seed,
    )
    client_indices = partition_with_plan(
        train_indices, labels[train_indices], plan, seed
    )
    test_client_indices = partition_with_plan(
        test_indices, labels[test_indices], plan, seed + 1
    )
    return (
        assemble_subsets(all_data, client_indices),
        assemble_subsets(all_data, test_client_indices),
        num_classes,
    )


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """为已组装的 ``Subset`` 创建 DataLoader。"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
