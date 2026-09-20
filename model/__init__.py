"""模型构建与注册。"""

from pathlib import Path

import torch

from .cnn import CNN


class Model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.extractor: torch.nn.Sequential = model.extractor
        self.classifier: torch.nn.Linear = model.classifier

    def forward(self, inputs: torch.Tensor, req_feat=False):
        feature: torch.Tensor = self.extractor(inputs)
        output: torch.Tensor = self.classifier(feature)
        if req_feat:
            return feature, output
        else:
            return output


def build_model(
    name: str,
    dataset: str,
    num_classes: int,
    feature_dim: int,
    pre_train: str | None = None,
) -> Model:
    """构建模型，并按需要加载本地预训练权重。"""
    if name == "cnn":
        base_model = CNN(1 if dataset == "mnist" else 3, num_classes, feature_dim)
    else:
        raise ValueError(f"不支持的模型：{name}")
    model = Model(base_model)
    if pre_train is not None:
        path = Path(pre_train)
        if not path.is_file():
            raise ValueError(f"{name} 无官方预训练权重；pre-train 须是本地权重")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model
