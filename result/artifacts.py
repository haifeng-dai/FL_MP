"""实验目录、日志、指标、checkpoint 与 SQLite 摘要。"""

import csv
import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """一次运行的文件路径。"""

    directory: Path
    log_file: Path
    metrics_file: Path
    checkpoint_dir: Path
    database: Path
    run_id: str


def create_run(args) -> RunPaths:
    """创建运行目录、写入参数并初始化日志和指标文件。"""
    arguments = vars(args).copy()
    encoded = json.dumps(arguments, sort_keys=True, ensure_ascii=False).encode()
    digest = hashlib.blake2s(encoded, digest_size=3).hexdigest()
    now = datetime.now(UTC)
    run_id = f"{now:%H%M%S}-{args.algo.lower()}-{digest}"
    directory = (
        Path(args.result_root) / f"{now:%Y}" / f"{now:%m}" / f"{now:%d}" / run_id
    )
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    (directory / "args.json").write_text(
        json.dumps(arguments, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths = RunPaths(
        directory=directory,
        log_file=directory / "run.log",
        metrics_file=directory / "metrics.csv",
        checkpoint_dir=checkpoints,
        database=Path(args.result_root) / "results.sqlite",
        run_id=run_id,
    )
    with paths.metrics_file.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(
            file, fieldnames=("round", "train_loss", "accuracy", "elapsed_seconds")
        ).writeheader()
    return paths


def configure_logging(paths: RunPaths, level: str) -> logging.Logger:
    """配置仅属于当前运行的日志记录器。"""
    logger = logging.getLogger(f"fl_mp.{paths.run_id}")
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.FileHandler(paths.log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s\n%(message)s"))
    logger.addHandler(handler)
    return logger


def append_metrics(paths: RunPaths, metrics: dict[str, float | int]) -> None:
    """追加一轮固定基础指标。"""
    with paths.metrics_file.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(
            file, fieldnames=("round", "train_loss", "accuracy", "elapsed_seconds")
        ).writerow(metrics)


def record_summary(
    paths: RunPaths,
    args,
    started: datetime,
    ended: datetime,
    status: str,
    max_accuracy: float,
    latest_accuracy: float,
) -> None:
    """在全局 SQLite 摘要库登记本次运行。"""
    paths.database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(paths.database) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, algorithm TEXT NOT NULL, started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL, status TEXT NOT NULL, max_accuracy REAL NOT NULL,
                latest_accuracy REAL NOT NULL, run_path TEXT NOT NULL, args_hash TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                paths.run_id,
                args.algo,
                started.isoformat(),
                ended.isoformat(),
                status,
                max_accuracy,
                latest_accuracy,
                str(paths.directory),
                paths.run_id.rsplit("-", 1)[1],
            ),
        )
