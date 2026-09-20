"""CNN 的表征接口与旧权重兼容性测试。"""

import torch

from algo.core import BaseClient
from model import build_model
from model.cnn import CNN


class ReferenceClient(BaseClient):
    """仅用于验证 BaseClient 参考模型构造逻辑的最小实现。"""

    def train(self, task):
        raise NotImplementedError


def test_cnn_exposes_extractor_and_classifier() -> None:
    """全局算法应能显式取得 feature_dim 表征与分类 logits。"""
    model = CNN(in_channels=3, num_classes=10, feature_dim=16)
    features = model.extractor(torch.randn(2, 3, 32, 32))

    assert features.shape == (2, 16)
    assert model.classifier(features).shape == (2, 10)


def test_build_model_loads_legacy_cnn_state(tmp_path) -> None:
    """旧版 features/classifier 序列权重应继续可被 pre-train 加载。"""
    model = CNN(in_channels=3, num_classes=10, feature_dim=16)
    legacy: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if name.startswith("extractor."):
            index, suffix = name.removeprefix("extractor.").split(".", maxsplit=1)
            if int(index) < 7:
                legacy[f"features.{index}.{suffix}"] = value
            else:
                legacy[f"classifier.0.{suffix}"] = value
        else:
            legacy["classifier.2." + name.removeprefix("classifier.")] = value
    path = tmp_path / "legacy_cnn.pt"
    torch.save(legacy, path)

    restored = build_model("cnn", "cifar10", 10, 16, str(path))
    assert all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in model.state_dict().items()
    )


def test_copy_model_rebuilds_frozen_reference() -> None:
    """参考模型必须独立重建、加载状态且不参与梯度计算。"""
    source = CNN(in_channels=3, num_classes=10, feature_dim=16)
    client = object.__new__(ReferenceClient)
    client.model_name = "cnn"
    client.dataset_name = "cifar10"
    client.num_classes = 10
    client.feature_dim = 16
    client.device = torch.device("cpu")

    reference = client.copy_model(source.state_dict())
    assert not reference.training
    assert not any(parameter.requires_grad for parameter in reference.parameters())
    assert all(
        torch.equal(value, reference.state_dict()[name])
        for name, value in source.state_dict().items()
    )
