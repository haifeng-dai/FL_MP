"""模型构建与注册。"""

from pathlib import Path

import torch

from .cnn import CNN


def _convert_legacy_cnn_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """将早期 ``features/classifier`` 分段键名转换为统一 CNN 接口。"""
    converted: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if name.startswith("features."):
            name = "extractor." + name.removeprefix("features.")
        elif name.startswith("classifier.0."):
            name = "extractor.7." + name.removeprefix("classifier.0.")
        elif name.startswith("classifier.2."):
            name = "classifier." + name.removeprefix("classifier.2.")
        converted[name] = value
    return converted


def build_model(
    name: str,
    dataset: str,
    num_classes: int,
    feature_dim: int,
    pre_train: str | None = None,
) -> torch.nn.Module:
    """构建模型，并按需要加载本地预训练权重。"""
    if name != "cnn":
        raise ValueError(f"不支持的模型：{name}")
    model = CNN(1 if dataset == "mnist" else 3, num_classes, feature_dim)
    if pre_train is not None:
        path = Path(pre_train)
        if not path.is_file():
            raise ValueError("CNN 不提供官方预训练权重；pre-train 必须是本地权重文件")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(_convert_legacy_cnn_state(state))
    return model
