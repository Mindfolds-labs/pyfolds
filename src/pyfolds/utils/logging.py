"""Configuração profissional de logging para PyFolds."""

import json
import logging
import logging.handlers
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Deque, Optional, Tuple, Union

__all__ = [
    "TRACE_LEVEL",
    "PyFoldsLogger",
    "StructuredFormatter",
    "CircularBufferFileHandler",
    "FixedLayoutFormatter",
    "next_log_path",
    "build_log_path",
    "get_logger",
    "setup_logging",
    "setup_run_logging",
    "trace",
]

TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")
logging.TRACE = TRACE_LEVEL


def trace(self, message, *args, **kwargs):
    """Método trace para logging (nível 5 - mais detalhado que DEBUG)."""
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)


logging.Logger.trace = trace


class StructuredFormatter(logging.Formatter):
    """JSON formatter para logs estruturados (ideal para produção)."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "context"):
            log_obj["context"] = record.context

        if hasattr(record, "metrics"):
            log_obj["metrics"] = record.metrics

        return json.dumps(log_obj)


class FixedLayoutFormatter(logging.Formatter):
    """Formatter de layout fixo (largura estável) para auditoria/debug."""

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created)
        date = dt.strftime("%Y%m%d")
        time_ = dt.strftime("%H%M%S") + f".{int(record.msecs):03d}"

        level = f"{record.levelname:<8}"[:8]
        logger_name = f"{record.name:<35}"[:35]
        file_line = f"{record.filename}:{record.lineno:<4}"[:14]

        msg = record.getMessage().replace("\n", "\\n").replace("\r", "\\r")
        return f"{date} {time_} | {level} | {logger_name} | {file_line} | {msg}"

class PyFoldsLogger:
    """Gerenciador singleton de logging para PyFolds."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_bootstrapped", False):
            return

        self._bootstrapped = True
        self._default_level = logging.INFO
        self._handlers = []
        self._loggers = {}
        self._initialized = False

        # Não configura handlers automaticamente no import.
        # A configuração deve ser explícita via setup()/setup_logging().

    def setup(
        self,
        level: Union[str, int] = "INFO",
        log_file: Optional[Union[str, Path]] = None,
        format_string: Optional[str] = None,
        module_levels: Optional[dict] = None,
        structured: bool = False,
        circular_buffer_lines: Optional[int] = None,
        console: bool = False,
        fixed_layout: bool = False,
    ):
        """Configura logging global e permite reconfiguração segura."""
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        self._default_level = level
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove handlers existentes para evitar duplicação e permitir reconfig.
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

        self._handlers = []

        if structured:
            formatter = StructuredFormatter()
        elif fixed_layout:
            formatter = FixedLayoutFormatter()
        else:
            if format_string is None:
                format_string = (
                    "%(asctime)s | %(name)-30s | %(levelname)-8s | "
                    "%(filename)s:%(lineno)d | %(message)s"
                )
            formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            self._handlers.append(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            if circular_buffer_lines is not None:
                file_handler = CircularBufferFileHandler(
                    log_path,
                    capacity_lines=circular_buffer_lines,
                    encoding="utf-8",
                )
            else:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_path, maxBytes=10485760, backupCount=5, encoding="utf-8"
                )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            self._handlers.append(file_handler)

        if module_levels:
            for module, lvl in module_levels.items():
                logging.getLogger(module).setLevel(getattr(logging, lvl.upper(), level))

        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)

        self._initialized = True
        logging.info(f"✅ Logging configurado - nível: {logging.getLevelName(level)}")

    def get_logger(self, name: str, level: Optional[int] = None) -> logging.Logger:
        """Retorna logger configurado."""
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)

        logger = self._loggers[name]
        logger.propagate = True
        logger.disabled = False
        if level is not None:
            logger.setLevel(level)
        return logger

    def add_file_handler(
        self,
        path: Union[str, Path],
        level: Optional[str] = None,
        max_bytes: int = 10485760,
        backup_count: int = 5,
    ):
        """Adiciona handler de arquivo adicional com rotação."""
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )

        if level:
            handler.setLevel(getattr(logging, level.upper()))

        root_logger = logging.getLogger()
        if root_logger.handlers:
            handler.setFormatter(root_logger.handlers[0].formatter)

        root_logger.addHandler(handler)
        self._handlers.append(handler)


logger_manager = PyFoldsLogger()


def get_logger(name: str) -> logging.Logger:
    """Função conveniência para obter logger."""
    return logger_manager.get_logger(name)




def setup_run_logging(
    app: str = "pyfolds",
    version: Optional[str] = None,
    log_dir: Union[str, Path] = "logs",
    level: Union[str, int] = "INFO",
    structured: bool = False,
    fixed_layout: bool = True,
    console: bool = False,
    circular_buffer_lines: Optional[int] = None,
) -> Tuple[logging.Logger, Path]:
    """Configuração recomendada para execução (treino/debug em produção).

    Cria automaticamente um arquivo incremental e configura logging para arquivo
    com terminal silencioso por padrão.

    Returns:
        Tupla ``(logger, log_path)``.
    """
    if version is None:
        try:
            import pyfolds as _pyfolds
            version = getattr(_pyfolds, "__version__", "unknown")
        except Exception:
            version = "unknown"

    log_path = next_log_path(log_dir=log_dir, app=app, version=version)
    logger_manager.setup(
        level=level,
        log_file=log_path,
        structured=structured,
        circular_buffer_lines=circular_buffer_lines,
        console=console,
        fixed_layout=fixed_layout,
    )
    logger = logger_manager.get_logger(app, level=level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))
    logger.info("Run logging configured: %s", log_path)
    return logger, log_path

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    structured: bool = False,
    circular_buffer_lines: Optional[int] = None,
    console: bool = False,
    fixed_layout: bool = False,
) -> logging.Logger:
    """Setup rápido de logging."""
    logger_manager.setup(
        level=level,
        log_file=log_file,
        structured=structured,
        circular_buffer_lines=circular_buffer_lines,
        console=console,
        fixed_layout=fixed_layout,
    )
    return logger_manager.get_logger("pyfolds")


class CircularBufferFileHandler(logging.Handler):
    """Handler TXT com buffer circular em número de linhas.

    Mantém somente as últimas ``capacity_lines`` mensagens no arquivo,
    sobrescrevendo-o a cada emissão para preservar ordem cronológica.

    Nota:
        Recomendado para debug e inspeção dos últimos eventos.
        Não é recomendado para treinos longos, pois cada ``emit``
        reescreve o arquivo inteiro (custo O(n) no tamanho do buffer).
    """

    def __init__(self, path: Union[str, Path], capacity_lines: int, encoding: str = "utf-8"):
        super().__init__()
        if capacity_lines <= 0:
            raise ValueError(
                f"circular_buffer_lines must be a positive integer, got {capacity_lines}"
            )

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.encoding = encoding
        self.capacity_lines = capacity_lines
        self._lock = Lock()
        self._buffer: Deque[str] = deque(maxlen=capacity_lines)
        self._line_count = 0

        if self.path.exists():
            previous_lines = self.path.read_text(encoding=self.encoding).splitlines()
            self._buffer.extend(previous_lines[-capacity_lines:])
            self._line_count = len(self._buffer)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            with self._lock:
                self._buffer.append(message)
                content = "\n".join(self._buffer)
                if content:
                    content += "\n"
                self.path.write_text(content, encoding=self.encoding)
                self._line_count = len(self._buffer)
        except Exception:
            self.handleError(record)


def next_log_path(log_dir: Union[str, Path], app: str, version: str) -> Path:
    """Gera caminho de log incremental: NNN_app_version_YYYYMMDD_HHMMSS.log."""
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)

    nums = []
    for path in directory.glob("*.log"):
        try:
            nums.append(int(path.name.split("_")[0]))
        except (ValueError, IndexError):
            continue

    next_idx = (max(nums) + 1) if nums else 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"{next_idx:03d}_{app}_{version}_{ts}.log"


def build_log_path(log_dir: Union[str, Path], app: str, version: str) -> Path:
    """Alias de compatibilidade para ``next_log_path``."""
    return next_log_path(log_dir=log_dir, app=app, version=version)
