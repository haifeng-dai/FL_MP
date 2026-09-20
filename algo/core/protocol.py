"""客户端任务、结果和训练钩子的公共协议。"""

import abc
import argparse
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ClientTask:
    """调度给客户端 Worker 的算法无关任务。"""

    client_id: int
    state: dict[str, torch.Tensor]
    seed: int
    payload: Any = None


@dataclass
class ClientResult:
    """客户端返回的独立 CPU 状态、基础指标和算法私有载荷。"""

    client_id: int
    state: dict[str, torch.Tensor]
    num_samples: int
    loss: float
    payload: Any = None


@dataclass(frozen=True)
class EvaluationTask:
    """调度给客户端 Worker 的个性化评估任务。"""

    client_id: int
    state: dict[str, torch.Tensor]


@dataclass(frozen=True)
class EvaluationResult:
    """客户端测试集上的正确预测数与样本数。"""

    client_id: int
    correct: int
    num_samples: int


class ClientHook(abc.ABC):
    """算法向通用 Worker 注入本地训练逻辑的扩展点。"""

    @abc.abstractmethod
    def __init__(
        self,
        args: argparse.Namespace,
        device: torch.device,
        num_classes: int,
    ) -> None:
        """构造客户端运行资源；实现不得保存整个 ``args``。"""
        raise NotImplementedError

    @abc.abstractmethod
    def set_data(self, train_sets: Any, test_sets: Any = None) -> None:
        """绑定训练数据及个性化算法可选的客户端测试数据。"""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, task: ClientTask) -> ClientResult:
        """执行一项客户端训练任务并返回独立 CPU 结果。"""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, task: EvaluationTask) -> EvaluationResult:
        """执行一项客户端私有测试集评估任务。"""
        raise NotImplementedError
