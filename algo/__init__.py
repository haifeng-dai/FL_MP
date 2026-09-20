"""联邦学习算法入口与实现。"""

import argparse
import importlib


def load_algorithm():
    """先读取 --algo，再加载对应算法模块。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--algo", required=True, help="算法名称，例如 fedavg")
    known, _ = parser.parse_known_args()
    try:
        return importlib.import_module(f"algo.{known.algo.lower()}")
    except ModuleNotFoundError as exc:
        if exc.name == f"algo.{known.algo.lower()}":
            raise ValueError(f"不支持的算法：{known.algo}") from exc
        raise
