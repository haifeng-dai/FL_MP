"""FL_MP 统一命令入口和通用运行编排。"""

import argparse
import copy
import traceback
from datetime import UTC, datetime
from types import ModuleType

import torch

from algo import load_algorithm
from algo.core.config import (
    add_shared_arguments,
    parse_trials,
    set_seed,
    validate_arguments,
)
from algo.core.decentralized import add_topology_arguments, validate_topology_arguments
from save import configure_logging, create_run, record_running, record_summary


def run_once(algorithm: ModuleType, args: argparse.Namespace) -> None:
    """以一个明确试次独立运行算法。"""
    torch.multiprocessing.set_sharing_strategy("file_system")
    set_seed(args.seed)
    started = datetime.now(UTC)
    paths = create_run(args)
    record_running(paths, args, started)
    logger = configure_logging(paths, args.log_level)
    args._startup_log_file = str(paths.log_file)
    args._startup_log_lock = torch.multiprocessing.get_context("spawn").Lock()
    try:
        server = algorithm.Server(args)
        server.fit(paths, logger)
    except Exception:
        ended = datetime.now(UTC)
        logger.exception("运行失败")
        record_summary(paths, args, started, ended, "failed", 0.0, 0.0)
        raise
    ended = datetime.now(UTC)
    record_summary(
        paths,
        args,
        started,
        ended,
        "completed",
        max(server.accuracies),
        server.accuracies[-1],
    )


def run_algorithm(algorithm: ModuleType) -> int:
    """解析参数并按框架级试次独立运行算法。"""
    parser = argparse.ArgumentParser(description=f"FL_MP {algorithm.__name__}")
    add_shared_arguments(parser)
    is_decentralized = algorithm.__name__.startswith("algo.dfl.")
    if is_decentralized:
        add_topology_arguments(parser)
    add_arguments = getattr(algorithm, "add_arguments", None)
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args()
    expected_name = algorithm.__name__.rsplit(".", maxsplit=1)[-1]
    if args.algo.lower() != expected_name:
        raise ValueError(f"算法模块 {expected_name} 与 --algo={args.algo} 不一致")
    validate_arguments(args)
    if is_decentralized:
        validate_topology_arguments(args)
    failed_trials: list[int] = []
    for trial in parse_trials(args):
        trial_args = copy.copy(args)
        trial_args.trial = trial
        print(f"\n============= 试次 {trial} =============", flush=True)
        try:
            run_once(algorithm, trial_args)
        except Exception:  # noqa: BLE001 - 单个试次失败不应中断其余指定试次
            failed_trials.append(trial)
            traceback.print_exc()
    if failed_trials:
        print("失败试次：" + ", ".join(map(str, failed_trials)), flush=True)
        return 1
    return 0


def main() -> int:
    """加载 ``--algo`` 指定的算法，并交由通用运行编排执行。"""
    return run_algorithm(load_algorithm())


if __name__ == "__main__":
    raise SystemExit(main())
