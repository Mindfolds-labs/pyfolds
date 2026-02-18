import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "run_pyfolds.py"


def _run(script_path: Path, *extra: str):
    return subprocess.run(
        [sys.executable, str(RUNNER), str(script_path), *extra],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )


def test_runner_logs_syntax_error(tmp_path: Path):
    script = tmp_path / "syntax_bad.py"
    script.write_text("def x(:\n    pass\n", encoding="utf-8")

    proc = _run(script, "--no-tee", "--log-dir", str(tmp_path), "--app-name", "pyfolds")
    assert proc.returncode != 0

    logs = sorted(tmp_path.glob("*.log"))
    assert logs
    content = logs[-1].read_text(encoding="utf-8")
    assert "SyntaxError" in content
    assert "[STDERR]" in content
    assert "EXIT CODE:" in content


def test_runner_logs_runtime_error_and_propagates_exit_code(tmp_path: Path):
    script = tmp_path / "runtime_bad.py"
    script.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    proc = _run(script, "--no-tee", "--log-dir", str(tmp_path), "--app-name", "pyfolds")
    assert proc.returncode == 1

    logs = sorted(tmp_path.glob("*.log"))
    assert logs
    content = logs[-1].read_text(encoding="utf-8")
    assert "RuntimeError: boom" in content
    assert "EXIT CODE: 1" in content


def test_runner_tee_prints_progress(tmp_path: Path):
    script = tmp_path / "ok.py"
    script.write_text("print('hello-runner')\n", encoding="utf-8")

    proc = _run(script, "--log-dir", str(tmp_path), "--app-name", "pyfolds")
    assert proc.returncode == 0
    assert "hello-runner" in proc.stdout

    logs = sorted(tmp_path.glob("*.log"))
    assert logs
    content = logs[-1].read_text(encoding="utf-8")
    assert "[STDOUT] hello-runner" in content
    assert "EXIT CODE: 0" in content


def test_runner_timeout_watchdog(tmp_path: Path):
    script = tmp_path / "hang.py"
    script.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")

    proc = _run(
        script,
        "--no-tee",
        "--log-dir",
        str(tmp_path),
        "--timeout-seconds",
        "0.5",
    )
    assert proc.returncode != 0

    logs = sorted(tmp_path.glob("*.log"))
    content = logs[-1].read_text(encoding="utf-8")
    assert "Timeout atingido" in content


def test_runner_retry_and_metrics_csv(tmp_path: Path):
    marker = tmp_path / "marker.txt"
    script = tmp_path / "flaky.py"
    script.write_text(
        "from pathlib import Path\n"
        f"marker=Path(r'{marker}')\n"
        "if not marker.exists():\n"
        "    marker.write_text('1')\n"
        "    raise RuntimeError('first-fail')\n"
        "print('ok-second')\n",
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.csv"

    proc = _run(
        script,
        "--no-tee",
        "--log-dir",
        str(tmp_path),
        "--max-retries",
        "1",
        "--metrics-csv",
        str(metrics),
    )
    assert proc.returncode == 0
    assert metrics.exists()

    csv_content = metrics.read_text(encoding="utf-8")
    assert "exit_code" in csv_content
    assert ",0," in csv_content
