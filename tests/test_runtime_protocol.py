"""通用 Worker 协议的无 GPU 单元测试。"""

import importlib

import torch
from torch.utils.data import TensorDataset

from algo.core import BaseServer
from algo.local import LocalClient
from algo.local import Server as LocalServer
from runtime import ClientResult, EvaluationResult, EvaluationTask, clone_state


class PflServerHarness(BaseServer):
    """用于验证个性化服务端基础行为的最小实现。"""

    pfl = True

    def apply_results(self, results):
        self.update_client_states(results)


class EvaluationPoolHarness:
    """模拟常驻 Worker 池返回的个性化评估结果。"""

    def evaluate(self, tasks):
        assert [task.client_id for task in tasks] == [0, 1]
        return {
            0: EvaluationResult(0, correct=1, num_samples=1),
            1: EvaluationResult(1, correct=0, num_samples=3),
        }


def test_clone_state_creates_independent_cpu_tensors() -> None:
    """状态复制必须与输入 Tensor 独立且不保留计算图。"""
    source = {"weight": torch.ones(2, requires_grad=True)}
    cloned = clone_state(source)

    assert cloned["weight"].device.type == "cpu"
    assert not cloned["weight"].requires_grad
    assert cloned["weight"].data_ptr() != source["weight"].data_ptr()


def test_personalized_server_uses_client_specific_task_states() -> None:
    """不同客户端任务必须携带各自的独立模型状态。"""
    server = object.__new__(PflServerHarness)
    server.client_states = {
        0: {"weight": torch.tensor([1.0])},
        1: {"weight": torch.tensor([2.0])},
    }
    server.selected = [0, 1]
    server.seed = 7
    server.accuracies = []

    tasks = server.build_tasks()
    assert tasks[0].state["weight"].item() == 1.0
    assert tasks[1].state["weight"].item() == 2.0
    assert tasks[0].state["weight"].data_ptr() != server.client_states[0]["weight"].data_ptr()


def test_training_states_hook_overrides_default_task_state() -> None:
    """需要独立下发模型的算法应只覆写状态钩子。"""
    server = object.__new__(PflServerHarness)
    server.client_states = {0: {"weight": torch.tensor([1.0])}}
    server.selected = [0]
    server.seed = 7
    server.training_states = lambda: {0: {"weight": torch.tensor([9.0])}}

    assert server.build_tasks()[0].state["weight"].item() == 9.0


def test_personalized_server_writes_selected_states_only() -> None:
    """未参与本轮训练的客户端状态必须保持不变。"""
    server = object.__new__(PflServerHarness)
    server.client_states = {
        0: {"weight": torch.tensor([1.0])},
        1: {"weight": torch.tensor([2.0])},
    }
    server.selected = [0]
    server.update_client_states(
        {
            0: ClientResult(0, {"weight": torch.tensor([3.0])}, 1, 0.0),
        }
    )

    assert server.client_states[0]["weight"].item() == 3.0
    assert server.client_states[1]["weight"].item() == 2.0


def test_personalized_evaluation_reports_macro_and_micro_accuracy() -> None:
    """个性化主指标为客户端 macro accuracy，同时保留 micro accuracy。"""
    server = object.__new__(PflServerHarness)
    server.client_states = {
        0: {"weight": torch.tensor([[1.0], [-1.0]])},
        1: {"weight": torch.tensor([[-1.0], [1.0]])},
    }
    server.test_sets = {
        0: TensorDataset(torch.tensor([[1.0]]), torch.tensor([0])),
        1: TensorDataset(
            torch.tensor([[1.0], [-1.0], [-1.0]]), torch.tensor([0, 1, 1])
        ),
    }
    server.pool = EvaluationPoolHarness()

    macro = server.evaluate_clients()
    assert macro == 50.0
    assert server.personalized_micro_accuracy == 25.0


def test_client_evaluation_returns_counts_without_model_state() -> None:
    """客户端评估应只回传正确数与样本数。"""
    client = object.__new__(LocalClient)
    client.device = torch.device("cpu")
    client.batch_size = 8
    client.model = torch.nn.Linear(1, 2, bias=False)
    state = {"weight": torch.tensor([[1.0], [-1.0]])}
    client.test_sets = {
        0: TensorDataset(torch.tensor([[1.0], [-1.0]]), torch.tensor([0, 1]))
    }

    result = client.evaluate(EvaluationTask(0, state))
    assert result == EvaluationResult(0, correct=2, num_samples=2)


def test_local_algorithm_uses_base_server_personalization_hooks() -> None:
    """Local 使用 BaseServer 的个性化状态和评估钩子。"""
    assert issubclass(LocalServer, BaseServer)
    assert LocalServer.client_class is LocalClient


def test_all_centralized_personalized_algorithms_expose_server() -> None:
    """所有已迁入的中心化个性化算法均应可由统一入口加载。"""
    names = (
        "fedala",
        "feddpc",
        "fedkd",
        "fedper",
        "fedproto",
        "fedrep",
        "fedsa",
        "fedtgp",
        "fml",
        "lgfedavg",
        "local",
    )
    for name in names:
        assert hasattr(importlib.import_module(f"algo.{name}"), "Server")


def test_all_decentralized_algorithms_expose_server() -> None:
    """去中心化算法也必须能由统一入口动态加载。"""
    names = (
        "dfedavgm", "dfedpgp", "dfedset", "dispfl", "efhc", "l2c", "pearfl", "proxyfl",
    )
    for name in names:
        assert hasattr(importlib.import_module(f"algo.{name}"), "Server")
