import logging
import time
from pathlib import Path

from pyfolds.utils.logging import CircularBufferFileHandler


def _build_record(level: int, message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="pyfolds.test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_circular_buffer_lazy_flush_by_interval(tmp_path: Path):
    path = tmp_path / "lazy.log"
    handler = CircularBufferFileHandler(path, capacity_lines=10, flush_interval_sec=0.2)
    handler.setFormatter(logging.Formatter("%(message)s"))

    handler.emit(_build_record(logging.INFO, "line-1"))
    assert not path.exists()

    time.sleep(0.25)
    handler.emit(_build_record(logging.INFO, "line-2"))

    assert path.read_text(encoding="utf-8").splitlines() == ["line-1", "line-2"]
    handler.close()


def test_circular_buffer_flushes_immediately_on_error(tmp_path: Path):
    path = tmp_path / "error.log"
    handler = CircularBufferFileHandler(path, capacity_lines=10, flush_interval_sec=60)
    handler.setFormatter(logging.Formatter("%(message)s"))

    handler.emit(_build_record(logging.INFO, "line-1"))
    assert not path.exists()

    handler.emit(_build_record(logging.ERROR, "line-2"))
    assert path.read_text(encoding="utf-8").splitlines() == ["line-1", "line-2"]
    handler.close()
