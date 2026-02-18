#!/usr/bin/env python3
"""Runner robusto para execução de scripts Python com logging completo.

Características:
- captura stdout/stderr em arquivo estruturado
- nome incremental com app/version/timestamp
- suporte a tee opcional para mostrar progresso no terminal
- retorno de exit code correto para PowerShell/CI
- retry opcional, watchdog por timeout e métricas CSV
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import io
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

_HEADER_LINE = "=" * 40


def _detect_pyfolds_version() -> str:
    """Tenta detectar versão instalada do pyfolds sem quebrar execução."""
    try:
        return importlib.metadata.version("pyfolds")
    except Exception:
        pass

    try:
        mod = importlib.import_module("pyfolds")
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _next_log_path(log_dir: Path, app_name: str, version: str) -> Path:
    """Gera log incremental: NNN_app_version_YYYYMMDD_HHMMSS.log."""
    log_dir.mkdir(parents=True, exist_ok=True)

    nums: list[int] = []
    for path in log_dir.glob("*.log"):
        try:
            nums.append(int(path.name.split("_")[0]))
        except (ValueError, IndexError):
            continue

    next_idx = (max(nums) + 1) if nums else 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"{next_idx:03d}_{app_name}_{version}_{ts}.log"


def _rotate_logs(log_dir: Path, keep: int) -> None:
    """Mantém apenas os `keep` logs mais recentes no diretório."""
    if keep <= 0:
        return
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in logs[keep:]:
        try:
            old.unlink()
        except OSError:
            continue


def _stream_reader(stream: io.TextIOBase, label: str, out_q: queue.Queue[tuple[str, str]]) -> None:
    """Lê stream linha a linha e envia para fila thread-safe."""
    try:
        for line in iter(stream.readline, ""):
            if line == "":
                break
            out_q.put((label, line))
    finally:
        stream.close()


def _gpu_monitor_worker(stop_evt: threading.Event, out_q: queue.Queue[tuple[str, str]], interval_s: float) -> None:
    """Monitora GPU via nvidia-smi (best effort)."""
    while not stop_evt.is_set():
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                for line in proc.stdout.strip().splitlines():
                    out_q.put(("GPU", line + "\n"))
            else:
                out_q.put(("RUNNER", "GPU monitor unavailable (nvidia-smi not ready)\n"))
                return
        except Exception:
            out_q.put(("RUNNER", "GPU monitor unavailable\n"))
            return

        stop_evt.wait(interval_s)


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    """Finaliza subprocesso com fallback kill."""
    if proc.poll() is not None:
        return

    if os.name == "nt":
        proc.terminate()
    else:
        proc.send_signal(signal.SIGTERM)

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _append_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    """Append de métricas CSV por execução."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "start",
                "end",
                "duration_s",
                "exit_code",
                "script",
                "log_path",
                "attempt",
                "version",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_script(
    script_path: str,
    args: list[str],
    tee: bool = True,
    log_dir: str = "logs",
    app_name: str = "pyfolds",
    timeout_seconds: float | None = None,
    max_retries: int = 0,
    metrics_csv: str | None = None,
    monitor_gpu: bool = False,
    checkpoint_path: str | None = None,
    resume_arg: str = "--resume-from",
) -> int:
    """Executa script alvo e salva stdout/stderr em log estruturado."""
    version = _detect_pyfolds_version()
    log_path = _next_log_path(Path(log_dir), app_name=app_name, version=version)

    script_abs = str(Path(script_path).resolve())
    base_args = list(args)

    attempts = max(1, max_retries + 1)
    final_exit_code = 1
    total_start_perf = time.perf_counter()
    start_ts = datetime.now()

    with log_path.open("w", encoding="utf-8") as logf:
        _write_header(logf, version=version, script=script_abs, command=[sys.executable, script_path, *base_args], start=start_ts)
        if tee:
            print(_HEADER_LINE)
            print("PYFOLDS EXECUTION RUN")
            print(f"Version: {version}")
            print(f"Script: {script_abs}")
            print(f"Log: {log_path}")
            print(_HEADER_LINE)

        for attempt in range(1, attempts + 1):
            command = [sys.executable, script_path, *base_args]
            if checkpoint_path and Path(checkpoint_path).exists() and resume_arg not in command:
                command.extend([resume_arg, checkpoint_path])

            logf.write(f"[RUNNER] Attempt {attempt}/{attempts}\n")
            logf.flush()

            try:
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except Exception:
                tb = traceback.format_exc()
                logf.write("[RUNNER] Falha ao iniciar subprocesso\n")
                logf.write(tb)
                final_exit_code = 1
                break

            assert proc.stdout is not None
            assert proc.stderr is not None

            out_q: queue.Queue[tuple[str, str]] = queue.Queue()
            t_out = threading.Thread(target=_stream_reader, args=(proc.stdout, "STDOUT", out_q), daemon=True)
            t_err = threading.Thread(target=_stream_reader, args=(proc.stderr, "STDERR", out_q), daemon=True)
            t_out.start()
            t_err.start()

            gpu_stop = threading.Event()
            gpu_thread: threading.Thread | None = None
            if monitor_gpu:
                gpu_thread = threading.Thread(
                    target=_gpu_monitor_worker,
                    args=(gpu_stop, out_q, 5.0),
                    daemon=True,
                )
                gpu_thread.start()

            attempt_start = time.perf_counter()
            timed_out = False
            while True:
                try:
                    label, line = out_q.get(timeout=0.1)
                    formatted = f"[{label}] {line.rstrip()}\n"
                    logf.write(formatted)
                    logf.flush()

                    if tee:
                        if label == "STDERR":
                            sys.stderr.write(line)
                            sys.stderr.flush()
                        elif label == "STDOUT":
                            sys.stdout.write(line)
                            sys.stdout.flush()
                        else:
                            sys.stdout.write(f"[{label}] {line}")
                            sys.stdout.flush()
                except queue.Empty:
                    if timeout_seconds is not None and (time.perf_counter() - attempt_start) > timeout_seconds:
                        timed_out = True
                        logf.write(f"[RUNNER] Timeout atingido ({timeout_seconds}s). Encerrando processo.\n")
                        logf.flush()
                        _terminate_process(proc)

                    if proc.poll() is not None and out_q.empty():
                        break

            gpu_stop.set()
            if gpu_thread is not None:
                gpu_thread.join(timeout=1)

            t_out.join(timeout=1)
            t_err.join(timeout=1)

            exit_code = int(proc.wait())
            if timed_out and exit_code == 0:
                exit_code = 124
            final_exit_code = exit_code

            if exit_code == 0:
                break

            if attempt < attempts:
                logf.write(f"[RUNNER] Retry após falha (exit_code={exit_code})\n")
                logf.flush()

        duration = time.perf_counter() - total_start_perf
        _write_footer(logf, exit_code=final_exit_code, duration=duration)

    if metrics_csv:
        _append_metrics_csv(
            Path(metrics_csv),
            {
                "start": start_ts.isoformat(),
                "end": datetime.now().isoformat(),
                "duration_s": f"{(time.perf_counter() - total_start_perf):.3f}",
                "exit_code": final_exit_code,
                "script": script_abs,
                "log_path": str(log_path),
                "attempt": attempts,
                "version": version,
            },
        )

    if tee:
        print(_HEADER_LINE)
        print(f"EXIT CODE: {final_exit_code}")
        print(f"DURATION: {(time.perf_counter() - total_start_perf):.3f} seconds")
        print(f"LOG FILE: {log_path}")
        print(_HEADER_LINE)

    _rotate_logs(Path(log_dir), keep=200)
    return final_exit_code


def _write_header(
    logf: TextIO,
    *,
    version: str,
    script: str,
    command: list[str],
    start: datetime,
) -> None:
    logf.write(f"{_HEADER_LINE}\n")
    logf.write("PYFOLDS EXECUTION RUN\n")
    logf.write(f"Version: {version}\n")
    logf.write(f"Script: {script}\n")
    logf.write(f"Command: {' '.join(command)}\n")
    logf.write(f"Start: {start.isoformat()}\n")
    logf.write(f"{_HEADER_LINE}\n")
    logf.flush()


def _write_footer(logf: TextIO, *, exit_code: int, duration: float) -> None:
    logf.write(f"{_HEADER_LINE}\n")
    logf.write(f"EXIT CODE: {exit_code}\n")
    logf.write(f"DURATION: {duration:.3f} seconds\n")
    logf.write(f"End: {datetime.now().isoformat()}\n")
    logf.write(f"{_HEADER_LINE}\n")
    logf.flush()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner robusto para scripts PyFolds")
    parser.add_argument("script", help="Script Python alvo")
    parser.add_argument("script_args", nargs="*", help="Argumentos do script")
    parser.add_argument("--no-tee", action="store_true", help="Não espelhar saída no terminal")
    parser.add_argument("--log-dir", default="logs", help="Diretório para logs")
    parser.add_argument("--app-name", default="pyfolds", help="Nome da aplicação no arquivo de log")
    parser.add_argument("--timeout-seconds", type=float, default=None, help="Watchdog timeout por execução")
    parser.add_argument("--max-retries", type=int, default=0, help="Número de retries automáticos")
    parser.add_argument("--metrics-csv", default=None, help="Arquivo CSV para métricas de execução")
    parser.add_argument("--monitor-gpu", action="store_true", help="Coleta periódica de métricas GPU (nvidia-smi)")
    parser.add_argument("--checkpoint-path", default=None, help="Checkpoint para retomar em retry")
    parser.add_argument("--resume-arg", default="--resume-from", help="Flag usada para passar checkpoint ao script")

    if hasattr(parser, "parse_intermixed_args"):
        return parser.parse_intermixed_args(argv)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv or sys.argv[1:])

    script = ns.script
    if not Path(script).exists():
        print(f"❌ Script não encontrado: {script}", file=sys.stderr)
        return 2

    return run_script(
        script_path=script,
        args=ns.script_args,
        tee=not ns.no_tee,
        log_dir=ns.log_dir,
        app_name=ns.app_name,
        timeout_seconds=ns.timeout_seconds,
        max_retries=max(0, ns.max_retries),
        metrics_csv=ns.metrics_csv,
        monitor_gpu=ns.monitor_gpu,
        checkpoint_path=ns.checkpoint_path,
        resume_arg=ns.resume_arg,
    )


if __name__ == "__main__":
    sys.exit(main())
