"""联邦训练中模型状态的复制工具。"""

import torch


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """复制为脱离计算图的独立 CPU 模型状态。"""
    return {name: value.detach().cpu().clone() for name, value in state.items()}
