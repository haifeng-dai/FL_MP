"""并行运行多组实验，并保证每组内的算法按顺序执行。"""

import argparse
import asyncio
import json
import shlex
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 直接在这里维护实验计划。各组同时运行，每组内的命令按列表顺序串行执行。
# common_args 是该组所有算法共享的实验条件，用于公平比较。
# commands 只填写算法名与该算法的特定参数；两组都会使用此处指定的全部 GPU。
GROUP_COMMANDS: dict[str, dict[str, str | list[str]]] = {
    "group_a": {
        "common_args": "--dataset cifar10 --partition dirichlet --dir 0.1 --devices 0:5,1:5,2:5,3:5 --service-device 0",
        "commands": [
            "--algo fedavg",
            "--algo feddyn --alpha-coef 0.01",
            "--algo moon --mu 0.01 --tau 0.5",
            "--algo scaffold --global-lr 1.0",
        ],
    },
    "group_b": {
        "common_args": "--dataset cifar10 --partition dirichlet --dir 0.1 --devices 0:5,1:5,2:5,3:5 --service-device 3",
        "commands": [
            "--algo fedprox --mu 0.01",
            "--algo fedproc",
            "--algo fedfm --mu 0.1",
            "--algo fedpln --lambda 80.0",
        ],
    },
}


@dataclass(frozen=True)
class Group:
    """一组共享实验条件且串行执行的算法命令。"""

    name: str
    common_args: list[str]
    commands: list[list[str]]


@dataclass(frozen=True)
class CommandResult:
    """一条算法命令的执行结果。"""

    group: str
    index: int
    command: list[str]
    log_file: str
    return_code: int
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    """解析批处理日志目录。"""
    parser = argparse.ArgumentParser(description="并行运行多组串行实验")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("results/batches"),
        help="批处理日志根目录",
    )
    return parser.parse_args()


def load_groups() -> list[Group]:
    """读取并验证脚本内嵌的实验计划。"""
    if not GROUP_COMMANDS:
        raise ValueError("GROUP_COMMANDS 至少需要定义一组实验")

    groups: list[Group] = []
    names: set[str] = set()
    for name, raw_group in GROUP_COMMANDS.items():
        if (
            not isinstance(name, str)
            or not name.replace("_", "").replace("-", "").isalnum()
            or name in names
        ):
            raise ValueError("组名必须唯一，且只能包含字母、数字、下划线或连字符")
        if not isinstance(raw_group, dict):
            raise TypeError(f"组 {name} 必须是字典")
        common_args_text = raw_group.get("common_args")
        command_texts = raw_group.get("commands")
        if not isinstance(common_args_text, str):
            raise TypeError(f"组 {name} 的 common_args 必须是命令字符串")
        if (
            not isinstance(command_texts, list)
            or not command_texts
            or not all(isinstance(command, str) for command in command_texts)
        ):
            raise ValueError(f"组 {name} 的命令必须是非空字符串列表")
        common_args = shlex.split(common_args_text)
        commands = [shlex.split(command_text) for command_text in command_texts]
        if any(not command for command in commands):
            raise ValueError(f"组 {name} 不能包含空命令")
        for command in commands:
            algorithm_name([*common_args, *command])
        groups.append(Group(name, common_args, commands))
        names.add(name)
    return groups


def algorithm_name(arguments: list[str]) -> str:
    """从命令参数中提取算法名，用于生成可读日志文件名。"""
    for index, argument in enumerate(arguments):
        if argument == "--algo" and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith("--algo="):
            return argument.removeprefix("--algo=")
    raise ValueError("每条命令必须通过 --algo 或 --algo=<名称> 指定算法")


async def run_group(group: Group, log_directory: Path) -> list[CommandResult]:
    """串行运行一组命令；失败的命令会记录后继续下一条。"""
    results: list[CommandResult] = []
    group_directory = log_directory / group.name
    group_directory.mkdir(parents=True, exist_ok=False)
    for index, command_args in enumerate(group.commands, start=1):
        arguments = [*group.common_args, *command_args]
        algorithm = algorithm_name(arguments)
        command = ["uv", "run", "main.py", *arguments]
        log_file = group_directory / f"{index:02d}-{algorithm}.log"
        print(f"[{group.name}] 开始 {algorithm}: {shlex.join(command)}", flush=True)
        started = perf_counter()
        with log_file.open("w", encoding="utf-8") as file:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=PROJECT_ROOT,
                stdout=file,
                stderr=asyncio.subprocess.STDOUT,
            )
            return_code = await process.wait()
        elapsed = perf_counter() - started
        status = "完成" if return_code == 0 else f"失败（退出码 {return_code}）"
        print(
            f"[{group.name}] {algorithm} {status}，耗时 {elapsed:.1f}s",
            flush=True,
        )
        results.append(
            CommandResult(
                group.name,
                index,
                command,
                str(log_file),
                return_code,
                elapsed,
            )
        )
    return results


async def run(groups: list[Group], log_directory: Path) -> list[CommandResult]:
    """并行执行多组串行实验。"""
    group_results = await asyncio.gather(
        *(run_group(group, log_directory) for group in groups)
    )
    return [result for results in group_results for result in results]


def main() -> int:
    """创建批处理目录、运行内嵌计划并写入汇总。"""
    args = parse_args()
    groups = load_groups()
    started_at = datetime.now().astimezone()
    log_directory = args.log_root / started_at.strftime("%Y/%m/%d/%H%M%S")
    log_directory.mkdir(parents=True, exist_ok=False)
    results = asyncio.run(run(groups, log_directory))
    summary_path = log_directory / "summary.json"
    summary_path.write_text(
        json.dumps(
            [asdict(result) for result in results], ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    failures = [result for result in results if result.return_code != 0]
    print(
        (
            f"批处理结束：成功 {len(results) - len(failures)}，失败 {len(failures)}；"
            f"汇总：{summary_path}"
        ),
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
