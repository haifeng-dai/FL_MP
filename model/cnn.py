"""FL_MP 的基础卷积分类模型。"""

import torch


class CNN(torch.nn.Module):
    """支持 28×28 MNIST 和 32×32 CIFAR 输入的卷积分类器。"""

    def __init__(self, in_channels: int, num_classes: int, feature_dim: int):
        super().__init__()
        self.extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, feature_dim),
            torch.nn.ReLU(inplace=True),
        )
        self.classifier = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """返回每个类别的未归一化 logits。"""
        return self.classifier(self.extractor(inputs))
