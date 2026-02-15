"""Configuração profissional de logging para PyFolds."""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

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

        # setup inicial com defaults; chamadas futuras de setup reconfiguram.
        self.setup()

    def setup(
        self,
        level: Union[str, int] = "INFO",
        log_file: Optional[Union[str, Path]] = None,
        format_string: Optional[str] = None,
        module_levels: Optional[dict] = None,
        structured: bool = False,
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
        else:
            if format_string is None:
                format_string = (
                    "%(asctime)s | %(name)-30s | %(levelname)-8s | "
                    "%(filename)s:%(lineno)d | %(message)s"
                )
            formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        self._handlers.append(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

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
            logger = logging.getLogger(name)
            if level is not None:
                logger.setLevel(level)
            self._loggers[name] = logger
        return self._loggers[name]

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


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    structured: bool = False,
) -> logging.Logger:
    """Setup rápido de logging."""
    logger_manager.setup(level=level, log_file=log_file, structured=structured)
    return logger_manager.get_logger("pyfolds")
