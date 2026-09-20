"""FL_MP 统一命令入口和通用运行编排。"""

import argparse
from datetime import UTC, datetime
from types import ModuleType

import torch

from algo import load_algorithm
from algo.core.config import add_shared_arguments, set_seed, validate_arguments
from result import configure_logging, create_run, record_summary


def run_algorithm(algorithm: ModuleType) -> int:
    """解析参数并运行已加载算法模块提供的 ``Server``。"""
    parser = argparse.ArgumentParser(description=f"FL_MP {algorithm.__name__}")
    add_shared_arguments(parser)
    add_arguments = getattr(algorithm, "add_arguments", None)
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args()
    expected_name = algorithm.__name__.rsplit(".", maxsplit=1)[-1]
    if args.algo.lower() != expected_name:
        raise ValueError(f"算法模块 {expected_name} 与 --algo={args.algo} 不一致")
    validate_arguments(args)
    torch.multiprocessing.set_sharing_strategy("file_system")
    set_seed(args.seed)
    paths = create_run(args)
    logger = configure_logging(paths, args.log_level)
    started = datetime.now(UTC)
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
    return 0


def main() -> int:
    """加载 ``--algo`` 指定的算法，并交由通用运行编排执行。"""
    return run_algorithm(load_algorithm())


if __name__ == "__main__":
    raise SystemExit(main())
