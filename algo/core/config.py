"""算法共享命令行参数与 CUDA 资源解析。"""

import argparse
import random

import torch


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """向算法参数解析器添加所有算法共用的运行参数。"""
    parser.add_argument("--algo", required=True)
    parser.add_argument(
        "--dataset", choices=("cifar10", "cifar100", "mnist"), default="cifar10"
    )
    parser.add_argument("--model", choices=("cnn",), default="cnn")
    parser.add_argument("--num-rounds", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--join-ratio", type=float, default=1.0)
    parser.add_argument(
        "--devices", default="0:1", help="每张 GPU 的 worker 数，例如 0:2,1:4"
    )
    parser.add_argument("--service-device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    trials = parser.add_mutually_exclusive_group()
    trials.add_argument(
        "--times", type=int, default=1, help="从第 1 次到第 N 次依次运行"
    )
    trials.add_argument("--trials", help="逗号分隔的试次编号，例如 1,3,5")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--partition", choices=("iid", "dirichlet", "pathological"), default="dirichlet"
    )
    parser.add_argument("--dir", type=float, default=0.1)
    parser.add_argument(
        "--classes-per-client",
        type=int,
        default=None,
        help="病态划分时每个客户端持有的不同类别数",
    )
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--pre-train", default=None)
    parser.add_argument("--check-round", type=int, default=50)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--result-root", default="results")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )


def validate_arguments(args: argparse.Namespace) -> None:
    """验证共享参数的基本范围与 CUDA 设备请求。"""
    if args.num_rounds < 1 or args.num_epochs < 1 or args.batch_size < 1:
        raise ValueError("num-rounds、num-epochs 和 batch-size 必须为正数")
    if args.num_clients < 1 or not 0 < args.join_ratio <= 1:
        raise ValueError("num-clients 必须为正数，join-ratio 必须在 (0, 1] 内")
    if not 0 < args.test_ratio < 1 or args.dir <= 0:
        raise ValueError("test-ratio 必须在 (0, 1) 内，dir 必须为正数")
    if args.check_round < 1:
        raise ValueError("check-round 必须为正数")
    if args.classes_per_client is not None and args.classes_per_client < 1:
        raise ValueError("classes-per-client 必须为正数")
    if args.times < 1:
        raise ValueError("times 必须为正数")
    parse_devices(args.devices)
    if not torch.cuda.is_available():
        raise RuntimeError("FL_MP 仅支持 NVIDIA CUDA GPU")
    if args.service_device >= torch.cuda.device_count():
        raise ValueError("service-device 不可见")


def parse_devices(spec: str) -> list[str]:
    """将 ``0:2,1:4`` 解析为 worker 使用的 CUDA 设备列表。"""
    devices: list[str] = []
    seen: set[int] = set()
    try:
        for item in spec.split(","):
            gpu_text, count_text = item.split(":", maxsplit=1)
            gpu, count = int(gpu_text), int(count_text)
            if gpu < 0 or count < 1 or gpu in seen:
                raise ValueError
            seen.add(gpu)
            devices.extend([f"cuda:{gpu}"] * count)
    except ValueError as exc:
        raise ValueError("devices 格式应为 0:2,1:4，且 GPU 不得重复") from exc
    if (
        not devices
        or max(int(device.split(":")[1]) for device in devices)
        >= torch.cuda.device_count()
    ):
        raise ValueError("请求的 worker GPU 不可见")
    return devices


def parse_trials(args: argparse.Namespace) -> list[int]:
    """将框架级试次配置解析为有序且不重复的正整数编号。"""
    if args.trials is None:
        return list(range(1, args.times + 1))
    try:
        trials = [int(item.strip()) for item in args.trials.split(",")]
    except ValueError as exc:
        raise ValueError("trials 必须是逗号分隔的正整数") from exc
    if not trials or any(trial < 1 for trial in trials):
        raise ValueError("trials 必须是逗号分隔的正整数")
    return list(dict.fromkeys(trials))


def set_seed(seed: int) -> None:
    """设置主进程可控随机源。Worker 使用任务下发的确定性 seed。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
