"""联邦学习算法入口与实现。"""

import argparse
import importlib
from pathlib import Path

_ALGORITHM_DIRS = ("gfl", "pfl", "dfl", "cfl")


def load_algorithm():
    """先读取 --algo，再按算法文件名在分类目录中加载实现。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--algo", required=True, help="算法名称，例如 fedavg")
    known, _ = parser.parse_known_args()
    algorithm_name = known.algo.lower()
    matches = [
        category
        for category in _ALGORITHM_DIRS
        if (Path(__file__).parent / category / f"{algorithm_name}.py").is_file()
    ]
    if not matches:
        raise ValueError(f"不支持的算法：{known.algo}")
    if len(matches) > 1:
        raise ValueError(f"算法名称 {known.algo} 在多个分类目录中重复：{matches}")
    return importlib.import_module(f"algo.{matches[0]}.{algorithm_name}")
