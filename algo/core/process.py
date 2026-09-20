"""算法无关的持久原生多进程客户端池。"""

import argparse
import traceback
from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.multiprocessing as mp

from .protocol import (
    ClientHook,
    ClientResult,
    ClientTask,
    EvaluationResult,
    EvaluationTask,
)


def _worker(
    worker_id: int,
    device_name: str,
    hook_type: type[ClientHook],
    args: argparse.Namespace,
    num_classes: int,
    train_sets: Mapping[int, Any],
    test_sets: Mapping[int, Any] | None,
    inbox: Any,
    outbox: Any,
) -> None:
    """创建钩子实例并串行执行调度给本 Worker 的任务。"""
    try:
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        hook = hook_type(args, device, num_classes)
        hook.set_data(train_sets, test_sets)
    except Exception:  # noqa: BLE001 -- 必须将 Worker 初始化异常传给主进程。
        outbox.put((worker_id, None, None, traceback.format_exc()))
        return

    while (task := inbox.get()) is not None:
        try:
            if isinstance(task, ClientTask):
                result = hook.run(task)
                _validate_training_result(task, result)
            else:
                result = hook.evaluate(task)
                _validate_evaluation_result(task, result)
            outbox.put((worker_id, task.client_id, result, None))
        except Exception:  # noqa: BLE001 -- 必须将算法钩子异常传给主进程。
            outbox.put((worker_id, task.client_id, None, traceback.format_exc()))
        finally:
            torch.cuda.empty_cache()


def _validate_training_result(task: ClientTask, result: ClientResult) -> None:
    """在 Worker 内尽早验证算法钩子的基础协议。"""
    if result.client_id != task.client_id:
        raise ValueError("ClientHook 返回的 client_id 与任务不一致")
    for tensor in result.state.values():
        if tensor.device.type != "cpu" or tensor.requires_grad:
            raise ValueError("ClientHook 必须返回独立的 CPU 模型状态")


def _validate_evaluation_result(
    task: EvaluationTask, result: EvaluationResult
) -> None:
    """验证评估结果的客户端身份和计数范围。"""
    if result.client_id != task.client_id:
        raise ValueError("ClientHook 返回的 client_id 与评估任务不一致")
    if result.correct < 0 or result.num_samples < 0 or result.correct > result.num_samples:
        raise ValueError("评估结果的正确数或样本数无效")


class PersistentClientPool:
    """复用 Worker 的 spawn 进程池；不包含任何算法训练逻辑。"""

    def __init__(
        self,
        devices: list[str],
        hook_type: type[ClientHook],
        args: argparse.Namespace,
        num_classes: int,
        train_sets: Mapping[int, Any],
        test_sets: Mapping[int, Any] | None,
    ):
        self.context = mp.get_context("spawn")
        self.inboxes = [self.context.Queue() for _ in devices]
        self.outbox = self.context.Queue()
        self.processes = [
            self.context.Process(
                target=_worker,
                args=(
                    worker_id,
                    device,
                    hook_type,
                    args,
                    num_classes,
                    train_sets,
                    test_sets,
                    self.inboxes[worker_id],
                    self.outbox,
                ),
            )
            for worker_id, device in enumerate(devices)
        ]
        for process in self.processes:
            process.start()

    def run(self, tasks: list[ClientTask]) -> dict[int, ClientResult]:
        """调度任务；任一 Worker 错误都会使本轮立即失败。"""
        return cast(dict[int, ClientResult], self._run(tasks, "训练"))

    def evaluate(
        self, tasks: list[EvaluationTask]
    ) -> dict[int, EvaluationResult]:
        """复用同一批 Worker 并发执行客户端私有测试集评估。"""
        return cast(dict[int, EvaluationResult], self._run(tasks, "评估"))

    def _run(self, tasks, action: str):
        """调度同一协议类型的任务；任一 Worker 错误都会立即失败。"""
        results: dict[int, Any] = {}
        next_task = 0
        active = 0
        for worker_id in range(min(len(self.inboxes), len(tasks))):
            self.inboxes[worker_id].put(tasks[next_task])
            next_task += 1
            active += 1
        while active:
            worker_id, client_id, result, error = self.outbox.get()
            active -= 1
            if error is not None:
                raise RuntimeError(f"worker {worker_id} {action}失败：\n{error}")
            results[client_id] = result
            if next_task < len(tasks):
                self.inboxes[worker_id].put(tasks[next_task])
                next_task += 1
                active += 1
        return results

    def close(self) -> None:
        """通知 Worker 退出并关闭全部 IPC 队列。"""
        for inbox in self.inboxes:
            inbox.put(None)
        for process in self.processes:
            process.join()
        for queue in [*self.inboxes, self.outbox]:
            queue.close()
