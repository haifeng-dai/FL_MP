"""基础进度条的无终端依赖测试。"""

from algo.core import ProgressBar, format_duration


def test_format_duration() -> None:
    """秒数应格式化为固定宽度的时分秒。"""
    assert format_duration(0) == "00:00:00"
    assert format_duration(65.4) == "00:01:05"
    assert format_duration(3_661) == "01:01:01"


def test_progress_bar_renders_extra_fields(capsys) -> None:
    """算法展示字段应由通用进度条统一拼接。"""
    progress = ProgressBar("fedavg", 1)
    progress.update(1.0, 50.0, 50.0, 1.0, 1.0, {"alpha": "0.500"})
    progress.close()
    assert "alpha=0.500" in capsys.readouterr().err
