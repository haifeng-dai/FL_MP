"""全局算法的纯 CPU 辅助逻辑测试。"""

import argparse

import torch

from algo.core import BaseServer
from algo.fedfm import add_arguments as add_fedfm_arguments
from algo.fedlsa import AnchorMapping, FedLSAModel, separation_loss
from algo.fedlsa import Server as FedLSAServer
from algo.fedper import aggregate_states
from algo.fedpln import PLN, aggregate_pln_states, dist_contrastive_loss
from algo.fedproc import aggregate_prototypes, prototype_loss
from algo.fedrep import add_arguments as add_fedrep_arguments


def test_aggregate_prototypes_uses_per_class_counts() -> None:
    """原型聚合必须按类别样本数加权，缺失类别保留旧原型。"""
    old = torch.tensor([[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]])
    merged = aggregate_prototypes(
        [
            {
                "prototypes": torch.tensor([[1.0, 1.0], [3.0, 3.0], [0.0, 0.0]]),
                "counts": torch.tensor([2, 1, 0]),
            },
            {
                "prototypes": torch.tensor([[5.0, 5.0], [0.0, 0.0], [4.0, 4.0]]),
                "counts": torch.tensor([1, 0, 3]),
            },
        ],
        old,
        by_count=True,
    )
    assert torch.equal(merged[0], torch.tensor([7.0 / 3.0, 7.0 / 3.0]))
    assert torch.equal(merged[1], torch.tensor([3.0, 3.0]))
    assert torch.equal(merged[2], torch.tensor([4.0, 4.0]))


def test_prototype_loss_is_finite_for_zero_prototypes() -> None:
    """首轮全零全局原型不应导致非有限损失。"""
    loss = prototype_loss(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.zeros(2, 2),
        torch.tensor([0, 1]),
    )
    assert torch.isfinite(loss)


def test_fedfm_accepts_prototype_anchor_loss() -> None:
    """FedFM 应提供原型对比损失作为锚点对齐模式。"""
    parser = argparse.ArgumentParser()
    add_fedfm_arguments(parser)
    assert parser.parse_args(["--anchor-loss", "prototype"]).anchor_loss == "prototype"


def test_pln_generates_class_prototypes_and_honors_fixed_embeddings() -> None:
    """PLN 应生成正确维度原型，并可冻结类别嵌入。"""
    pln = PLN(3, width=4, feature_dim=2, depth=1, fixed=True, init_emb=1)
    assert pln(torch.arange(3)).shape == (3, 2)
    assert not pln.embeddings.weight.requires_grad


def test_pln_state_aggregation_and_distance_loss() -> None:
    """PLN 状态按权重平均，距离原型损失保持有限。"""
    states = [{"weight": torch.tensor([1.0])}, {"weight": torch.tensor([5.0])}]
    merged = aggregate_pln_states(states, [0.25, 0.75])
    assert torch.equal(merged["weight"], torch.tensor([4.0]))
    loss = dist_contrastive_loss(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0]),
    )
    assert torch.isfinite(loss)


def test_fedlsa_normalizes_features_and_generates_normalized_anchors() -> None:
    """FedLSA 表征和锚点均应为单位范数。"""
    base = torch.nn.Module()
    base.extractor = torch.nn.Linear(2, 3)
    base.classifier = torch.nn.Linear(3, 2)
    model = FedLSAModel(base)
    features = model.extractor(torch.randn(4, 2))
    anchors = AnchorMapping(3)(torch.randn(2, 3))

    assert torch.allclose(features.norm(dim=1), torch.ones(4), atol=1e-5)
    assert torch.allclose(anchors.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.isfinite(separation_loss(anchors, tau=0.1))


def test_fedlsa_uses_global_model_tasks() -> None:
    """FedLSA 每轮应广播全局模型，而非客户端历史模型。"""
    assert FedLSAServer.build_tasks is BaseServer.build_tasks
    assert not FedLSAServer.pfl


def test_fedper_aggregates_only_extractor_states() -> None:
    """FedPer 应按样本权重聚合共享特征提取器。"""
    aggregate = aggregate_states(
        [
            {"weight": torch.tensor([1.0]), "counter": torch.tensor(2)},
            {"weight": torch.tensor([5.0]), "counter": torch.tensor(8)},
        ],
        [0.25, 0.75],
    )
    assert torch.equal(aggregate["weight"], torch.tensor([4.0]))
    assert torch.equal(aggregate["counter"], torch.tensor(2))


def test_fedrep_accepts_head_training_epochs() -> None:
    """FedRep 应独立配置分类头训练阶段的轮数。"""
    parser = argparse.ArgumentParser()
    add_fedrep_arguments(parser)
    assert parser.parse_args(["--epochs-head", "3"]).epochs_head == 3
