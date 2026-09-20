"""浏览 FL_MP 已完成实验的摘要和准确率曲线。"""

import argparse
import csv
import json
import shutil
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "results" / "results.sqlite"
BASELINE_FIELDS = (
    "dataset",
    "model",
    "partition",
    "dir",
    "classes_per_client",
    "num_clients",
    "join_ratio",
    "num_rounds",
    "num_epochs",
    "batch_size",
    "lr",
    "momentum",
    "weight_decay",
    "feature_dim",
    "test_ratio",
    "seed",
    "pre_train",
    "adj_type",
    "edge_p",
    "k_small_world",
    "m_scale_free",
)
RUNTIME_FIELDS = {
    "algo",
    "data_root",
    "devices",
    "service_device",
    "log_level",
    "result_root",
    "times",
    "trials",
    "trial",
}
COMMON_ARGUMENT_DEFAULTS = {
    "dataset": "cifar10",
    "model": "cnn",
    "num_rounds": 100,
    "num_epochs": 10,
    "lr": 0.01,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "batch_size": 64,
    "num_clients": 20,
    "join_ratio": 1.0,
    "devices": "0:1",
    "service_device": 0,
    "seed": 42,
    "times": 1,
    "trials": None,
    "trial": 1,
    "test_ratio": 0.2,
    "partition": "dirichlet",
    "dir": 0.1,
    "classes_per_client": None,
    "feature_dim": 512,
    "pre_train": None,
    "check_round": 50,
    "data_root": "datasets",
    "result_root": "results",
    "adj_type": "ring",
    "edge_p": 0.2,
    "k_small_world": 4,
    "m_scale_free": 2,
    "log_level": "INFO",
}
SYMBOLS = ("●", "◆", "■", "▲", "●", "◆", "■", "▲")


@dataclass(frozen=True)
class RunRecord:
    """SQLite 摘要与真实运行目录中的参数。"""

    run_id: str
    algorithm: str
    status: str
    max_accuracy: float
    latest_accuracy: float
    started_at: str
    ended_at: str
    run_path: Path
    args: dict[str, Any] | None
    issue: str | None = None


@dataclass(frozen=True)
class Curve:
    """一条用于终端预览的准确率曲线。"""

    label: str
    run: RunRecord
    points: list[tuple[int, float]]


def parse_args() -> argparse.Namespace:
    """解析结果浏览命令。"""
    parser = argparse.ArgumentParser(description="浏览 FL_MP 实验结果")
    parser.add_argument(
        "command", nargs="?", choices=("list", "show", "compare", "aggregate")
    )
    parser.add_argument("run_id", nargs="?", help="show 命令要查看的运行 ID")
    parser.add_argument(
        "--database", type=Path, default=DEFAULT_DATABASE, help="SQLite 结果库路径"
    )
    parser.add_argument("--algo", help="逗号分隔的算法名称")
    parser.add_argument(
        "--status",
        choices=("running", "completed", "failed", "all"),
        default="completed",
        help="list 命令的运行状态筛选",
    )
    parser.add_argument("--limit", type=int, help="list 命令最多显示的记录数")
    parser.add_argument("--run", help="compare 命令使用的逗号分隔运行 ID")
    parser.add_argument("--vary", help="compare 单算法时变化的参数名")
    parser.add_argument("--trial", type=int, help="筛选单个试次编号")
    parser.add_argument("--trials", help="逗号分隔的试次编号，例如 1,3,5")
    args = parser.parse_args()
    args.command = args.command or "list"
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须为正数")
    if args.command == "show" and args.run_id is None:
        parser.error("show 命令需要提供运行 ID")
    if args.command != "show" and args.run_id is not None:
        parser.error("只有 show 命令接受位置运行 ID")
    if args.command == "compare" and not args.run and not args.algo:
        parser.error("compare 命令需要 --run 或 --algo")
    if args.vary and args.command != "compare":
        parser.error("--vary 仅适用于 compare 命令")
    if args.run and args.command != "compare":
        parser.error("--run 仅适用于 compare 命令")
    if args.run and args.vary:
        parser.error("--run 与 --vary 不能同时使用")
    if args.trial is not None and args.trial < 1:
        parser.error("--trial 必须为正数")
    if args.trial is not None and args.trials:
        parser.error("--trial 与 --trials 不能同时使用")
    if args.command == "aggregate" and not args.algo:
        parser.error("aggregate 命令需要 --algo")
    if args.command == "aggregate" and args.run:
        parser.error("aggregate 不支持 --run")
    return args


def parse_csv(value: str | None) -> list[str]:
    """解析逗号分隔参数并去重。"""
    if value is None:
        return []
    values = [item.strip() for item in value.split(",") if item.strip()]
    return list(dict.fromkeys(values))


def parse_trials(value: str | None) -> list[int] | None:
    """解析逗号分隔的正整数试次编号。"""
    if value is None:
        return None
    try:
        trials = [int(item) for item in parse_csv(value)]
    except ValueError as exc:
        raise ValueError("trials 必须是逗号分隔的正整数") from exc
    if not trials or any(trial < 1 for trial in trials):
        raise ValueError("trials 必须是逗号分隔的正整数")
    return trials


def resolve_run_path(path: str, root: Path = PROJECT_ROOT) -> Path:
    """将 SQLite 中的相对运行路径解析到项目根目录。"""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def load_args(run_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """读取运行参数；目录损坏时保留摘要记录并返回原因。"""
    path = run_path / "args.json"
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return None, "缺少 args.json"
    except json.JSONDecodeError:
        return None, "args.json 格式无效"
    if not isinstance(data, dict):
        return None, "args.json 顶层必须是对象"
    return data, None


def load_runs(
    database: Path, algorithms: list[str] | None = None, status: str = "completed"
) -> list[RunRecord]:
    """从 SQLite 读取运行摘要，并关联真实目录的参数文件。"""
    if not database.is_file():
        raise ValueError(f"找不到 SQLite 结果库：{database}")
    clauses: list[str] = []
    parameters: list[str] = []
    if status != "all":
        clauses.append("status = ?")
        parameters.append(status)
    if algorithms:
        placeholders = ", ".join("?" for _ in algorithms)
        clauses.append(f"algorithm IN ({placeholders})")
        parameters.extend(algorithms)
    where = "" if not clauses else " WHERE " + " AND ".join(clauses)
    query = (
        "SELECT run_id, algorithm, status, max_accuracy, latest_accuracy, "
        "started_at, ended_at, run_path FROM runs"
        f"{where} ORDER BY started_at DESC"
    )
    with sqlite3.connect(database) as connection:
        rows = connection.execute(query, parameters).fetchall()
    records: list[RunRecord] = []
    for row in rows:
        run_path = resolve_run_path(row[7])
        args, issue = load_args(run_path)
        if not run_path.is_dir():
            issue = "运行目录不存在"
        records.append(
            RunRecord(
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                run_path,
                args,
                issue,
            )
        )
    return records


def format_duration(started_at: str, ended_at: str) -> str:
    """格式化 SQLite 记录的运行时长。"""
    try:
        elapsed = datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)
        seconds = max(0, round(elapsed.total_seconds()))
    except ValueError:
        return "-"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def argument(record: RunRecord, name: str) -> Any:
    """读取单个运行参数；缺失参数显示为短横线。"""
    return "-" if record.args is None else record.args.get(name, "-")


def required_argument(record: RunRecord, name: str) -> Any:
    """读取比较操作所需的参数，并明确拒绝损坏的运行目录。"""
    if record.args is None or name not in record.args:
        raise ValueError(f"{record.run_id} 缺少参数 {name}")
    return record.args[name]


def normalized_arguments(record: RunRecord) -> dict[str, Any] | None:
    """补齐旧运行中未记录的共享参数默认值。"""
    if record.args is None:
        return None
    return {**COMMON_ARGUMENT_DEFAULTS, **record.args}


def trial_number(record: RunRecord) -> int | None:
    """读取运行试次；旧运行默认视作第 1 次。"""
    args = normalized_arguments(record)
    if args is None:
        return None
    trial = args["trial"]
    return trial if isinstance(trial, int) and trial > 0 else None


def filter_trials(records: list[RunRecord], trials: list[int] | None) -> list[RunRecord]:
    """按试次编号筛选记录。"""
    if trials is None:
        return records
    selected = set(trials)
    return [record for record in records if trial_number(record) in selected]


def dataset_partition(record: RunRecord) -> str:
    """紧凑显示数据集及数据划分参数。"""
    dataset = argument(record, "dataset")
    partition = argument(record, "partition")
    if partition == "dirichlet":
        partition = f"Dir({argument(record, 'dir')})"
    elif partition == "pathological":
        partition = f"病态({argument(record, 'classes_per_client')})"
    elif partition == "iid":
        partition = "IID"
    return f"{dataset} / {partition}"


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    """输出无依赖的 Unicode 表格。"""
    values = [[str(value) for value in row] for row in rows]
    widths = [display_width(header) for header in headers]
    for row in values:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], display_width(value))
    rule = "-+-".join("-" * width for width in widths)
    print(
        " | ".join(
            pad_display(header, width)
            for header, width in zip(headers, widths, strict=True)
        )
    )
    print(rule)
    for row in values:
        print(
            " | ".join(
                pad_display(value, width)
                for value, width in zip(row, widths, strict=True)
            )
        )


def display_width(value: str) -> int:
    """计算 CJK 宽字符在等宽终端中的显示宽度。"""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in value)


def pad_display(value: str, width: int) -> str:
    """按显示宽度而不是 Python 字符数填充表格单元格。"""
    return value + " " * (width - display_width(value))


def list_runs(records: list[RunRecord], limit: int | None) -> None:
    """显示运行摘要表。"""
    shown = records if limit is None else records[:limit]
    rows = [
        [
            record.run_id,
            record.algorithm,
            record.status,
            dataset_partition(record),
            argument(record, "num_clients"),
            argument(record, "num_rounds"),
            trial_number(record) or "-",
            f"{record.max_accuracy:.2f} / {record.latest_accuracy:.2f}%",
            format_duration(record.started_at, record.ended_at),
            record.started_at.replace("T", " ")[:19],
        ]
        for record in shown
    ]
    if not rows:
        print("没有符合条件的运行。")
        return
    print_table(
        [
            "运行 ID",
            "算法",
            "状态",
            "数据集 / 划分",
            "客户端",
            "轮数",
            "试次",
            "最佳 / 最终 acc",
            "时长",
            "开始时间",
        ],
        rows,
    )
    issues = [record for record in shown if record.issue]
    if issues:
        print("\n目录检查：")
        for record in issues:
            print(f"- {record.run_id}: {record.issue}")


def read_metrics(record: RunRecord) -> list[tuple[int, float]]:
    """读取并验证逐轮准确率指标。"""
    path = record.run_path / "metrics.csv"
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or not {"round", "accuracy"} <= set(
                reader.fieldnames
            ):
                raise ValueError("metrics.csv 缺少 round 或 accuracy 列")
            points = [(int(row["round"]), float(row["accuracy"])) for row in reader]
    except FileNotFoundError as exc:
        raise ValueError(f"{record.run_id} 缺少 metrics.csv") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{record.run_id} 的 metrics.csv 格式无效：{exc}") from exc
    if not points:
        raise ValueError(f"{record.run_id} 没有可用的逐轮指标")
    return sorted(points)


def print_run(record: RunRecord) -> None:
    """显示单次运行的摘要、参数与准确率曲线。"""
    print(f"运行 ID：{record.run_id}")
    print(f"算法：{record.algorithm}    状态：{record.status}")
    print(
        f"最佳 acc：{record.max_accuracy:.2f}%    最终 acc：{record.latest_accuracy:.2f}%"
    )
    print(f"目录：{record.run_path}")
    if record.issue:
        print(f"目录检查：{record.issue}")
    if record.args is not None:
        print("\n参数：")
        for name, value in sorted(record.args.items()):
            print(f"  {name} = {value}")
    try:
        points = read_metrics(record)
    except ValueError as exc:
        print(f"\n无法预览准确率曲线：{exc}")
        return
    print("\n准确率曲线：")
    print(render_accuracy_chart([Curve(record.algorithm, record, points)]))


def baseline_signature(
    record: RunRecord, ignored: set[str] | None = None
) -> tuple[tuple[str, str], ...] | None:
    """生成用于判断科研实验可比性的基础参数签名。"""
    args = normalized_arguments(record)
    if args is None:
        return None
    ignored = ignored or set()
    values: list[tuple[str, str]] = []
    for name in BASELINE_FIELDS:
        if name in ignored:
            continue
        values.append(
            (name, json.dumps(args[name], sort_keys=True, ensure_ascii=False))
        )
    return tuple(values)


def baseline_description(record: RunRecord) -> str:
    """以简短字段描述一组基础实验配置。"""
    return ", ".join(
        f"{name}={argument(record, name)}"
        for name in ("dataset", "partition", "num_clients", "num_rounds", "seed")
    )


def select_latest_comparable(
    records: list[RunRecord], algorithms: list[str]
) -> list[RunRecord]:
    """为每个算法选择同一基础实验签名下的最新完成运行。"""
    grouped: dict[tuple[tuple[str, str], ...], dict[str, list[RunRecord]]] = {}
    requested = set(algorithms)
    for record in records:
        if record.algorithm not in requested or record.status != "completed":
            continue
        signature = baseline_signature(record)
        if signature is not None:
            grouped.setdefault(signature, {}).setdefault(record.algorithm, []).append(
                record
            )
    candidates = [
        (signature, by_algorithm)
        for signature, by_algorithm in grouped.items()
        if requested <= set(by_algorithm)
    ]
    if not candidates:
        descriptions = list(
            dict.fromkeys(
                baseline_description(record)
                for record in records
                if record.algorithm in requested and record.status == "completed"
            )
        )
        message = "找不到同时包含指定算法的可比完成运行"
        if descriptions:
            message += "。可用基础配置：" + "；".join(descriptions)
        raise ValueError(message)
    _, selected_group = max(
        candidates,
        key=lambda item: max(
            record.started_at for values in item[1].values() for record in values
        ),
    )
    return [
        max(selected_group[algorithm], key=lambda record: record.started_at)
        for algorithm in algorithms
    ]


def select_parameter_comparison(
    records: list[RunRecord], algorithm: str, vary: str
) -> list[RunRecord]:
    """选择单算法中仅指定参数不同的一组最新完成运行。"""
    grouped: dict[tuple[tuple[str, str], ...], list[RunRecord]] = {}
    for record in records:
        if record.algorithm != algorithm or record.status != "completed":
            continue
        if record.args is None or vary not in record.args:
            continue
        signature = parameter_signature(record, vary)
        grouped.setdefault(signature, []).append(record)
    candidates = []
    for signature, values in grouped.items():
        by_value: dict[str, list[RunRecord]] = {}
        for record in values:
            by_value.setdefault(
                json.dumps(required_argument(record, vary), sort_keys=True), []
            ).append(record)
        if len(by_value) >= 2:
            candidates.append((signature, by_value))
    if not candidates:
        raise ValueError(f"找不到 {algorithm} 中仅 {vary} 不同的至少两条完成运行")
    _, selected_group = max(
        candidates,
        key=lambda item: max(
            record.started_at for values in item[1].values() for record in values
        ),
    )
    selected = [
        max(values, key=lambda record: record.started_at)
        for values in selected_group.values()
    ]
    return sorted(
        selected,
        key=lambda record: json.dumps(required_argument(record, vary), sort_keys=True),
    )


def parameter_signature(record: RunRecord, vary: str) -> tuple[tuple[str, str], ...]:
    """生成单算法参数对比的完整控制变量签名。"""
    args = normalized_arguments(record)
    assert args is not None
    return tuple(
        (name, json.dumps(value, sort_keys=True, ensure_ascii=False))
        for name, value in sorted(args.items())
        if name not in RUNTIME_FIELDS and name != vary
    )


def print_baseline_differences(records: list[RunRecord]) -> None:
    """显示显式指定运行之间不同的基础参数。"""
    differences: list[str] = []
    for name in BASELINE_FIELDS:
        values = {
            json.dumps(args.get(name), sort_keys=True)
            for record in records
            if (args := normalized_arguments(record)) is not None
        }
        if len(values) > 1:
            differences.append(name)
    if differences:
        print("基础参数差异：" + ", ".join(differences))


def compare_runs(
    records: list[RunRecord], vary: str | None = None, explicit: bool = False
) -> None:
    """输出比较摘要与多曲线 Unicode 图。"""
    curves: list[Curve] = []
    rows: list[list[Any]] = []
    for record in records:
        try:
            points = read_metrics(record)
        except ValueError as exc:
            print(f"跳过 {record.run_id}：{exc}")
            continue
        prefix = record.algorithm if vary is None else f"{vary}={required_argument(record, vary)}"
        label = f"{prefix} ({record.run_id})"
        curves.append(Curve(label, record, points))
        rows.append(
            [
                label,
                f"{record.max_accuracy:.2f}%",
                f"{record.latest_accuracy:.2f}%",
                len(points),
            ]
        )
    if not curves:
        raise ValueError("没有可用于比较的有效准确率曲线")
    print_table(["运行", "最佳 acc", "最终 acc", "记录轮数"], rows)
    if explicit:
        print_baseline_differences(records)
    print("\n准确率曲线：")
    print(render_accuracy_chart(curves))


def select_trials_for_aggregate(
    records: list[RunRecord], algorithms: list[str], trials: list[int] | None
) -> dict[str, list[RunRecord]]:
    """选择同一基础配置下各算法每个试次的最新完成运行。"""
    requested = set(algorithms)
    grouped: dict[
        tuple[tuple[str, str], ...], dict[str, dict[int, list[RunRecord]]]
    ] = {}
    for record in records:
        trial = trial_number(record)
        signature = baseline_signature(record)
        if (
            record.status != "completed"
            or record.algorithm not in requested
            or trial is None
            or signature is None
        ):
            continue
        grouped.setdefault(signature, {}).setdefault(record.algorithm, {}).setdefault(
            trial, []
        ).append(record)

    candidates: list[tuple[dict[str, dict[int, list[RunRecord]]], list[int]]] = []
    for by_algorithm in grouped.values():
        if not requested <= set(by_algorithm):
            continue
        available = set.intersection(
            *(set(by_algorithm[algorithm]) for algorithm in algorithms)
        )
        selected_trials = trials or sorted(available)
        if len(selected_trials) >= 2 and set(selected_trials) <= available:
            candidates.append((by_algorithm, selected_trials))
    if not candidates:
        requested_text = "全部共有试次" if trials is None else ",".join(map(str, trials))
        raise ValueError(
            f"找不到覆盖试次 {requested_text} 的同配置完成运行（聚合至少需要两次）"
        )
    by_algorithm, selected_trials = max(
        candidates,
        key=lambda item: max(
            record.started_at
            for algorithm in algorithms
            for trial in item[1]
            for record in item[0][algorithm][trial]
        ),
    )
    return {
        algorithm: [
            max(by_algorithm[algorithm][trial], key=lambda record: record.started_at)
            for trial in selected_trials
        ]
        for algorithm in algorithms
    }


def mean_curve(records: list[RunRecord]) -> list[tuple[int, float]]:
    """按共同轮次计算多次运行的平均准确率曲线。"""
    metrics = [dict(read_metrics(record)) for record in records]
    rounds = set.intersection(*(set(points) for points in metrics))
    if not rounds:
        raise ValueError("指定试次没有共同的有效轮次")
    return [(round_index, mean(points[round_index] for points in metrics)) for round_index in sorted(rounds)]


def aggregate_runs(grouped: dict[str, list[RunRecord]]) -> None:
    """输出固定 seed 下多次运行的终端统计与平均准确率曲线。"""
    curves: list[Curve] = []
    rows: list[list[Any]] = []
    for algorithm, records in grouped.items():
        latest = [record.latest_accuracy for record in records]
        maximum = [record.max_accuracy for record in records]
        trials = ",".join(str(trial_number(record)) for record in records)
        rows.append(
            [
                algorithm,
                trials,
                f"{mean(latest):.2f}% ± {stdev(latest):.2f}%",
                f"{mean(maximum):.2f}% ± {stdev(maximum):.2f}%",
            ]
        )
        curves.append(Curve(f"{algorithm} 平均", records[0], mean_curve(records)))
    print_table(["算法", "试次", "最终 acc", "最佳 acc"], rows)
    print("\n平均准确率曲线：")
    print(render_accuracy_chart(curves))


def render_accuracy_chart(
    curves: list[Curve], width: int | None = None, height: int = 12
) -> str:
    """渲染无依赖的多条准确率 Unicode 散点折线预览。"""
    if not curves:
        return "没有曲线。"
    width = width or max(30, min(90, shutil.get_terminal_size((100, 24)).columns - 16))
    points = [point for curve in curves for point in curve.points]
    x_min, x_max = min(point[0] for point in points), max(point[0] for point in points)
    y_min, y_max = min(point[1] for point in points), max(point[1] for point in points)
    padding = max(0.5, (y_max - y_min) * 0.1)
    y_min, y_max = max(0.0, y_min - padding), min(100.0, y_max + padding)
    if y_max - y_min < 1.0:
        y_min, y_max = max(0.0, y_min - 0.5), min(100.0, y_max + 0.5)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for index, curve in enumerate(curves):
        symbol = SYMBOLS[index % len(SYMBOLS)]
        for round_index, accuracy in curve.points:
            column = (
                0
                if x_max == x_min
                else round((round_index - x_min) * (width - 1) / (x_max - x_min))
            )
            row = round((y_max - accuracy) * (height - 1) / (y_max - y_min))
            grid[row][column] = symbol if grid[row][column] == " " else "*"
    lines = []
    for row, values in enumerate(grid):
        value = y_max - row * (y_max - y_min) / (height - 1)
        lines.append(f"{value:6.2f} │" + "".join(values))
    lines.append("       └" + "─" * width)
    lines.append(
        f"        r{x_min}"
        + " " * max(1, width - len(str(x_min)) - len(str(x_max)) - 1)
        + f"r{x_max}"
    )
    lines.append(
        "图例："
        + "  ".join(
            f"{SYMBOLS[index % len(SYMBOLS)]} {curve.label}"
            for index, curve in enumerate(curves)
        )
    )
    return "\n".join(lines)


def main() -> int:
    """执行列表、单次查看或比较命令。"""
    args = parse_args()
    try:
        algorithms = parse_csv(args.algo)
        requested_trials = [args.trial] if args.trial is not None else parse_trials(args.trials)
        records = load_runs(args.database, algorithms or None, args.status)
        records = filter_trials(records, requested_trials)
        if args.command == "list":
            list_runs(records, args.limit)
            return 0
        if args.command == "show":
            selected = next(
                (record for record in records if record.run_id == args.run_id), None
            )
            if selected is None:
                records = load_runs(args.database, status="all")
                selected = next(
                    (record for record in records if record.run_id == args.run_id), None
                )
            if selected is None:
                raise ValueError(f"找不到运行：{args.run_id}")
            print_run(selected)
            return 0
        if args.command == "aggregate":
            selected = select_trials_for_aggregate(
                records, algorithms, requested_trials
            )
            aggregate_runs(selected)
            return 0
        if args.run:
            requested = parse_csv(args.run)
            all_records = load_runs(args.database, status="all")
            indexed = {record.run_id: record for record in all_records}
            missing = [run_id for run_id in requested if run_id not in indexed]
            if missing:
                raise ValueError("找不到运行：" + ", ".join(missing))
            compare_runs([indexed[run_id] for run_id in requested], explicit=True)
            return 0
        if args.vary:
            if len(algorithms) != 1:
                raise ValueError("--vary 需要且只能指定一个 --algo")
            selected = select_parameter_comparison(records, algorithms[0], args.vary)
            compare_runs(selected, args.vary)
            return 0
        selected = select_latest_comparable(records, algorithms)
        compare_runs(selected)
        return 0
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
