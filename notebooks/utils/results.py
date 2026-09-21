"""FL_MP 实验结果 notebook 的共享读取、筛选、聚合与绘图工具。"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import matplotlib.pyplot as plt

RUNTIME_ARGUMENTS = {
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
REQUIRED_METRIC_COLUMNS = {
    "round",
    "train_loss",
    "accuracy",
    "elapsed_seconds",
}


@dataclass
class Run:
    """一次实验的完整索引；大文件仅保留路径，按需加载。"""

    run_id: str
    path: Path
    algorithm: str
    args: dict[str, Any]
    metrics: list[dict[str, Any]]
    metric_columns: tuple[str, ...]
    status: str = "unknown"
    started_at: str = ""
    ended_at: str = ""
    max_accuracy: float | None = None
    latest_accuracy: float | None = None
    checkpoints: tuple[Path, ...] = field(default_factory=tuple)
    log_path: Path | None = None

    @property
    def trial(self) -> int:
        """返回试次编号；未记录 trial 的旧实验视为第 1 次。"""
        value = self.args.get("trial", 1)
        return value if isinstance(value, int) and value > 0 else 1

    def series(
        self, metric: str, x: str = "round"
    ) -> tuple[list[float], list[float]]:
        """提取一条数值序列，并支持累计耗时派生指标。"""
        if metric == "cumulative_elapsed_seconds":
            xs, elapsed = self.series("elapsed_seconds", x)
            total = 0.0
            cumulative = []
            for value in elapsed:
                total += value
                cumulative.append(total)
            return xs, cumulative
        if x not in self.metric_columns:
            raise KeyError(f"{self.run_id} 的 metrics.csv 不包含横坐标列 {x!r}。")
        if metric not in self.metric_columns:
            raise KeyError(f"{self.run_id} 的 metrics.csv 不包含指标列 {metric!r}。")
        points = [
            (row.get(x), row.get(metric))
            for row in self.metrics
            if is_number(row.get(x)) and is_number(row.get(metric))
        ]
        if not points:
            raise ValueError(f"{self.run_id} 的指标 {metric!r} 没有有效数值。")
        return [float(point[0]) for point in points], [float(point[1]) for point in points]


def configure_plots() -> None:
    """设置适合 notebook 的统一 Matplotlib 风格。"""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def find_project_root(start: Path | None = None) -> Path:
    """从当前目录向上定位 FL_MP 项目根目录。"""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "algo").is_dir():
            return candidate
    raise FileNotFoundError("无法定位 FL_MP 项目根目录。")


def algorithm_names(directory: Path) -> set[str]:
    """从算法包自动发现算法名，新增算法文件后无需修改 notebook。"""
    if not directory.is_dir():
        raise FileNotFoundError(f"算法目录不存在：{directory}")
    names = {
        path.stem.lower()
        for path in directory.glob("*.py")
        if path.stem not in {"__init__", "test"}
    }
    if not names:
        raise ValueError(f"算法目录中没有可用算法文件：{directory}")
    return names


def is_number(value: Any) -> bool:
    """判断值是否为有限实数，并排除 bool。"""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _convert_cell(value: str | None) -> Any:
    if value is None or value.strip() == "":
        return None
    text = value.strip()
    try:
        return int(text)
    except ValueError:
        try:
            number = float(text)
            return number if math.isfinite(number) else text
        except ValueError:
            return text


def _read_metrics(path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"指标文件不存在：{path}")
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            columns = tuple(reader.fieldnames or ())
            if not columns:
                raise ValueError(f"指标文件没有表头：{path}")
            if len(columns) != len(set(columns)):
                raise ValueError(f"指标文件存在重复列名：{path}")
            missing = REQUIRED_METRIC_COLUMNS - set(columns)
            if missing:
                raise ValueError(
                    f"指标文件 {path} 缺少必需列："
                    + ", ".join(sorted(missing))
                )
            rows = [
                {name: _convert_cell(value) for name, value in row.items()}
                for row in reader
            ]
    except (OSError, csv.Error, UnicodeError) as exc:
        raise ValueError(f"无法读取指标文件 {path}：{exc}") from exc
    if not rows:
        raise ValueError(f"指标文件没有数据行：{path}")
    seen_rounds: set[int] = set()
    for line_number, row in enumerate(rows, start=2):
        for name in columns:
            if not is_number(row.get(name)):
                raise ValueError(
                    f"指标文件 {path} 第 {line_number} 行的 {name} 不是有限数值："
                    f"{row.get(name)!r}"
                )
        round_index = row["round"]
        if not isinstance(round_index, int) or round_index < 1:
            raise ValueError(
                f"指标文件 {path} 第 {line_number} 行的 round 必须是正整数。"
            )
        if round_index in seen_rounds:
            raise ValueError(f"指标文件 {path} 存在重复轮次：{round_index}")
        seen_rounds.add(round_index)
        expected_round = line_number - 1
        if round_index != expected_round:
            raise ValueError(
                f"指标文件 {path} 第 {line_number} 行轮次不连续："
                f"期望 {expected_round}，实际 {round_index}。"
            )
        if not 0 <= row["accuracy"] <= 100:
            raise ValueError(
                f"指标文件 {path} 第 {line_number} 行的 accuracy 超出 [0, 100]。"
            )
        if row["elapsed_seconds"] < 0:
            raise ValueError(
                f"指标文件 {path} 第 {line_number} 行的 elapsed_seconds 不能为负数。"
            )
    return rows, columns


def _database_rows(database: Path) -> list[dict[str, Any]]:
    """严格读取数据库中的运行索引；数据库是唯一数据入口。"""
    if not database.is_file():
        raise FileNotFoundError(f"SQLite 结果库不存在：{database}")
    try:
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM runs").fetchall()
    except (OSError, sqlite3.Error) as exc:
        raise ValueError(f"SQLite 结果库无法读取：{database}：{exc}") from exc
    if not rows:
        raise ValueError(f"SQLite 结果库中没有运行记录：{database}")
    required = {
        "run_id",
        "algorithm",
        "status",
        "started_at",
        "ended_at",
        "max_accuracy",
        "latest_accuracy",
        "run_path",
    }
    columns = set(rows[0].keys())
    missing = required - columns
    if missing:
        raise ValueError(
            f"SQLite 结果库 {database} 的 runs 表缺少字段："
            + ", ".join(sorted(missing))
        )
    return [dict(row) for row in rows]


def discover_runs(
    project_root: Path | None = None,
    result_root: Path | None = None,
    database: Path | None = None,
    algorithms: Iterable[str] | None = None,
    run_ids: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
) -> list[Run]:
    """以 SQLite 为唯一索引，严格读取指定算法的运行结果。"""
    project_root = (project_root or find_project_root()).resolve()
    result_root = (result_root or project_root / "results").resolve()
    database = database or result_root / "results.sqlite"
    records = _database_rows(database)
    requested = None if algorithms is None else {name.lower() for name in algorithms}
    if requested is not None and not requested:
        raise ValueError("algorithms 不能为空集合。")
    requested_ids = None if run_ids is None else set(run_ids)
    if requested_ids is not None and not requested_ids:
        raise ValueError("run_ids 不能为空集合。")
    if requested_ids is not None:
        available_ids = {str(record.get("run_id")) for record in records}
        missing_ids = requested_ids - available_ids
        if missing_ids:
            raise ValueError(
                "SQLite 结果库中找不到 run_id：" + ", ".join(sorted(missing_ids))
            )
    requested_statuses = None if statuses is None else set(statuses)
    if requested_statuses is not None and not requested_statuses:
        raise ValueError("statuses 不能为空集合。")
    runs: list[Run] = []
    for record in records:
        run_id = record.get("run_id")
        algorithm_value = record.get("algorithm")
        raw_path = record.get("run_path")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"SQLite 记录的 run_id 无效：{run_id!r}")
        if not isinstance(algorithm_value, str) or not algorithm_value.strip():
            raise ValueError(f"SQLite 记录 {run_id} 的 algorithm 无效。")
        algorithm = algorithm_value.lower()
        if requested is not None and algorithm not in requested:
            continue
        if requested_ids is not None and run_id not in requested_ids:
            continue
        if requested_statuses is not None and record.get("status") not in requested_statuses:
            continue
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"SQLite 记录 {run_id} 的 run_path 无效。")
        stored_path = Path(raw_path)
        run_path = (
            stored_path.resolve()
            if stored_path.is_absolute()
            else (project_root / stored_path).resolve()
        )
        if not run_path.is_dir():
            raise FileNotFoundError(
                f"SQLite 记录 {run_id} 指向的运行目录不存在：{run_path}"
            )
        args_path = run_path / "args.json"
        if not args_path.is_file():
            raise FileNotFoundError(f"参数文件不存在：{args_path}")
        try:
            args = json.loads(args_path.read_text(encoding="utf-8"))
            if not isinstance(args, dict):
                raise TypeError("顶层不是 JSON 对象")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"参数文件无效：{args_path}：{exc}") from exc
        args_algorithm = args.get("algo")
        if not isinstance(args_algorithm, str) or not args_algorithm.strip():
            raise ValueError(f"参数文件缺少非空字符串 algo：{args_path}")
        if args_algorithm.lower() != algorithm:
            raise ValueError(
                f"算法不一致：SQLite 记录 {run_id} 为 {algorithm_value!r}，"
                f"但 {args_path} 中为 {args_algorithm!r}。"
            )
        metrics, columns = _read_metrics(run_path / "metrics.csv")
        log_path = run_path / "run.log"
        if not log_path.is_file():
            raise FileNotFoundError(f"运行日志不存在：{log_path}")
        checkpoint_dir = run_path / "checkpoints"
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"checkpoint 目录不存在：{checkpoint_dir}")
        accuracies = [
            float(row["accuracy"])
            for row in metrics
            if is_number(row.get("accuracy"))
        ]
        max_accuracy = record.get("max_accuracy")
        latest_accuracy = record.get("latest_accuracy")
        runs.append(
            Run(
                run_id=run_id,
                path=run_path,
                algorithm=algorithm,
                args=args,
                metrics=metrics,
                metric_columns=columns,
                status=str(record["status"]),
                started_at=str(record["started_at"]),
                ended_at=str(record["ended_at"]),
                max_accuracy=(
                    float(max_accuracy)
                    if max_accuracy is not None
                    else (max(accuracies) if accuracies else None)
                ),
                latest_accuracy=(
                    float(latest_accuracy)
                    if latest_accuracy is not None
                    else (accuracies[-1] if accuracies else None)
                ),
                checkpoints=tuple(sorted(checkpoint_dir.glob("*.pt"))),
                log_path=log_path,
            )
        )
    if not runs:
        scope = "全部算法" if requested is None else ", ".join(sorted(requested))
        raise FileNotFoundError(
            f"没有找到实验结果；算法范围：{scope}；"
            f"状态范围：{requested_statuses or '全部'}"
        )
    return sorted(runs, key=lambda run: (run.started_at, run.run_id), reverse=True)


def family_runs(runs: Sequence[Run], names: Iterable[str]) -> list[Run]:
    """按算法名集合筛选一个算法族。"""
    algorithms = {name.lower() for name in names}
    selected = [run for run in runs if run.algorithm in algorithms]
    if not selected:
        raise ValueError("没有找到该算法族的任何实验结果。")
    return selected


def _matches(actual: Any, expected: Any) -> bool:
    if callable(expected):
        return bool(expected(actual))
    if isinstance(expected, (list, tuple, set, frozenset)):
        return actual in expected
    return actual == expected


def select_runs(runs: Sequence[Run], **filters: Any) -> list[Run]:
    """按运行属性或任意实验参数筛选运行。"""
    selected = []
    for run in runs:
        values = {
            "algorithm": run.algorithm,
            "algo": run.algorithm,
            "status": run.status,
            "run_id": run.run_id,
            "trial": run.trial,
            **run.args,
        }
        if all(_matches(values.get(name), expected) for name, expected in filters.items()):
            selected.append(run)
    if not selected:
        description = ", ".join(f"{name}={value!r}" for name, value in filters.items())
        raise ValueError(f"没有符合筛选条件的实验结果：{description or '无筛选条件'}")
    return selected


def runs_table(runs: Sequence[Run], limit: int | None = 30) -> list[dict[str, Any]]:
    """生成适合 notebook 直接显示的运行概览。"""
    if not runs:
        raise ValueError("没有可显示的实验结果。")
    shown = list(runs) if limit is None else list(runs)[:limit]
    return [
        {
            "run_id": run.run_id,
            "algorithm": run.algorithm,
            "status": run.status,
            "dataset": run.args.get("dataset"),
            "partition": run.args.get("partition"),
            "topology": run.args.get("adj_type"),
            "trial": run.trial,
            "rounds": len(run.metrics),
            "best_acc": run.max_accuracy,
            "final_acc": run.latest_accuracy,
            "checkpoints": len(run.checkpoints),
        }
        for run in shown
    ]


def differing_arguments(
    runs: Sequence[Run], ignore: Iterable[str] = RUNTIME_ARGUMENTS
) -> dict[str, list[Any]]:
    """返回一组运行中取值不一致的参数。"""
    ignored = set(ignore)
    names = set().union(*(run.args for run in runs)) if runs else set()
    differences = {}
    for name in sorted(names - ignored):
        encoded = {
            json.dumps(run.args.get(name), sort_keys=True, ensure_ascii=False)
            for run in runs
        }
        if len(encoded) > 1:
            differences[name] = [json.loads(value) for value in sorted(encoded)]
    return differences


def numeric_metrics(run: Run, x: str = "round") -> list[str]:
    """列出一次运行的全部可绘制指标。"""
    metrics = [
        name
        for name in run.metric_columns
        if name != x and any(is_number(row.get(name)) for row in run.metrics)
    ]
    if "elapsed_seconds" in metrics:
        metrics.append("cumulative_elapsed_seconds")
    return metrics


def get_run(run_or_id: Run | str, runs: Sequence[Run]) -> Run:
    """将运行对象或 ID 解析成唯一运行。"""
    if isinstance(run_or_id, Run):
        return run_or_id
    matches = [run for run in runs if run.run_id == run_or_id]
    if len(matches) != 1:
        raise ValueError(f"运行 {run_or_id!r} 匹配数量为 {len(matches)}。")
    return matches[0]


def plot_run(
    run_or_id: Run | str,
    runs: Sequence[Run],
    metrics: Sequence[str] | None = None,
    x: str = "round",
):
    """绘制一次运行的多个指标。"""
    run = get_run(run_or_id, runs)
    selected_metrics = list(metrics or numeric_metrics(run, x))
    if not selected_metrics:
        raise ValueError(f"{run.run_id} 没有可绘制的数值指标。")
    series = [(metric, *run.series(metric, x)) for metric in selected_metrics]
    columns = min(2, len(selected_metrics))
    rows = math.ceil(len(selected_metrics) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(6.2 * columns, 3.6 * rows), squeeze=False
    )
    for axis, (metric, xs, ys) in zip(axes.flat, series):
        axis.plot(xs, ys, linewidth=1.8)
        axis.set(title=metric, xlabel=x, ylabel=metric)
    for axis in list(axes.flat)[len(selected_metrics) :]:
        axis.set_visible(False)
    fig.suptitle(f"{run.algorithm} · {run.run_id}", y=1.01)
    fig.tight_layout()
    return fig, axes


def aggregate_series(
    runs: Sequence[Run], metric: str, x: str = "round"
) -> tuple[list[float], list[float], list[float]]:
    """按横坐标聚合多次运行的均值和样本标准差。"""
    by_x: dict[float, list[float]] = defaultdict(list)
    for run in runs:
        xs, ys = run.series(metric, x)
        for x_value, y_value in zip(xs, ys):
            by_x[x_value].append(y_value)
    coordinates = sorted(by_x)
    means = [fmean(by_x[value]) for value in coordinates]
    deviations = [
        stdev(by_x[value]) if len(by_x[value]) > 1 else 0.0
        for value in coordinates
    ]
    return coordinates, means, deviations


def _draw_groups(
    groups: dict[str, list[Run]], metric: str, title: str, show_std: bool
):
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for label, group in groups.items():
        xs, means, deviations = aggregate_series(group, metric)
        if not xs:
            continue
        line, = axis.plot(xs, means, linewidth=2, label=f"{label} (n={len(group)})")
        if show_std and len(group) > 1:
            axis.fill_between(
                xs,
                [mean - deviation for mean, deviation in zip(means, deviations)],
                [mean + deviation for mean, deviation in zip(means, deviations)],
                color=line.get_color(),
                alpha=0.15,
            )
    if not axis.lines:
        plt.close(fig)
        raise ValueError(f"匹配运行中没有指标 {metric!r}。")
    axis.set(title=title, xlabel="round", ylabel=metric)
    axis.legend()
    fig.tight_layout()
    return fig, axis


def plot_algorithm_runs(
    runs: Sequence[Run],
    algorithm: str,
    metric: str = "accuracy",
    *,
    filters: dict[str, Any] | None = None,
):
    """分别绘制一个算法的所有匹配运行。"""
    selected = select_runs(runs, algorithm=algorithm, **(filters or {}))
    if not selected:
        raise ValueError(f"没有找到 {algorithm} 的匹配运行。")
    fig, axis = plt.subplots(figsize=(9, 5))
    for run in reversed(selected):
        xs, ys = run.series(metric)
        if xs:
            axis.plot(xs, ys, alpha=0.8, label=f"{run.run_id} (trial={run.trial})")
    if not axis.lines:
        plt.close(fig)
        raise ValueError(f"匹配运行中没有指标 {metric!r}。")
    axis.set(title=f"{algorithm} · {metric}", xlabel="round", ylabel=metric)
    axis.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, axis, selected


def plot_algorithm_comparison(
    runs: Sequence[Run],
    algorithms: Sequence[str],
    metric: str = "accuracy",
    *,
    filters: dict[str, Any] | None = None,
    show_std: bool = True,
):
    """比较多个算法；每个算法内部按 trial 聚合。"""
    selected = select_runs(runs, algorithm=set(algorithms), **(filters or {}))
    groups = {
        algorithm: [run for run in selected if run.algorithm == algorithm]
        for algorithm in algorithms
    }
    missing = [algorithm for algorithm, group in groups.items() if not group]
    if missing:
        raise ValueError("没有匹配运行的算法：" + ", ".join(missing))
    fig, axis = _draw_groups(groups, metric, f"algorithm comparison · {metric}", show_std)
    differences = differing_arguments(selected)
    if differences:
        print("提醒：匹配运行中仍存在以下非运行时参数差异：")
        print(json.dumps(differences, ensure_ascii=False, indent=2))
    return fig, axis, selected


def plot_parameter_comparison(
    runs: Sequence[Run],
    algorithm: str,
    vary: str | Sequence[str],
    metric: str = "accuracy",
    *,
    filters: dict[str, Any] | None = None,
    show_std: bool = True,
):
    """比较单算法的一项或多项参数组合。"""
    names = [vary] if isinstance(vary, str) else list(vary)
    if not names:
        raise ValueError("vary 至少需要一个参数名。")
    selected = [
        run
        for run in select_runs(runs, algorithm=algorithm, **(filters or {}))
        if all(name in run.args for name in names)
    ]
    raw_groups: dict[tuple[Any, ...], list[Run]] = defaultdict(list)
    for run in selected:
        raw_groups[tuple(run.args[name] for name in names)].append(run)
    if len(raw_groups) < 2:
        raise ValueError(f"{algorithm} 在当前条件下不足两个参数组合：{names}")
    groups = {
        ", ".join(f"{name}={value}" for name, value in zip(names, key)): group
        for key, group in sorted(raw_groups.items(), key=lambda item: str(item[0]))
    }
    fig, axis = _draw_groups(
        groups, metric, f"{algorithm} parameter comparison · {metric}", show_std
    )
    remaining = differing_arguments(selected, RUNTIME_ARGUMENTS | set(names))
    if remaining:
        print("提醒：除对比参数外，以下参数仍有差异：")
        print(json.dumps(remaining, ensure_ascii=False, indent=2))
    return fig, axis, groups


def show_run_details(run_or_id: Run | str, runs: Sequence[Run]) -> dict[str, Any]:
    """显示一次运行的参数、指标列和原始文件位置。"""
    run = get_run(run_or_id, runs)
    return {
        "run_id": run.run_id,
        "algorithm": run.algorithm,
        "status": run.status,
        "path": str(run.path),
        "args": run.args,
        "metric_columns": run.metric_columns,
        "metric_rows": len(run.metrics),
        "checkpoints": [str(path) for path in run.checkpoints],
        "log_path": str(run.log_path) if run.log_path else None,
    }


def read_log(run_or_id: Run | str, runs: Sequence[Run], tail: int | None = 100) -> str:
    """读取完整日志或最后若干行。"""
    run = get_run(run_or_id, runs)
    if run.log_path is None:
        raise FileNotFoundError(f"{run.run_id} 没有 run.log。")
    lines = run.log_path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines if tail is None else lines[-tail:])


def load_checkpoint(
    run_or_id: Run | str,
    runs: Sequence[Run],
    round_index: int | None = None,
) -> dict[str, Any]:
    """按需将 checkpoint 安全映射到 CPU。"""
    import torch

    run = get_run(run_or_id, runs)
    if not run.checkpoints:
        raise FileNotFoundError(f"{run.run_id} 没有 checkpoint。")
    if round_index is None:
        candidates = []
        for path in run.checkpoints:
            try:
                checkpoint_round = int(path.stem.rsplit("_", 1)[-1])
            except ValueError as exc:
                raise ValueError(f"checkpoint 文件名不符合 round_N.pt：{path}") from exc
            if checkpoint_round < 1:
                raise ValueError(f"checkpoint 轮次必须为正整数：{path}")
            candidates.append((checkpoint_round, path))
        path = max(candidates)[1]
    else:
        path = run.path / "checkpoints" / f"round_{round_index}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)
