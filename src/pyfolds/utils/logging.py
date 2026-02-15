"""Configuração profissional de logging para PyFolds."""

import logging
import sys
from typing import Optional, Union
from pathlib import Path

# Níveis de log personalizados
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

def trace(self, message, *args, **kwargs):
    """Método trace para logging."""
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)

logging.Logger.trace = trace

class PyFoldsLogger:
    """Gerenciador de logging para PyFolds."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._loggers = {}
        self._default_level = logging.INFO
        self._handlers = []
    
    def setup(self,
              level: Union[str, int] = "INFO",
              log_file: Optional[Union[str, Path]] = None,
              format_string: Optional[str] = None,
              module_levels: Optional[dict] = None):
        """
        Configura logging global.
        
        Args:
            level: Nível padrão (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)
            log_file: Arquivo para salvar logs (opcional)
            format_string: Formato personalizado
            module_levels: Dict com níveis específicos por módulo
        """
        # Converte string para nível
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        
        self._default_level = level
        
        # Formato padrão
        if format_string is None:
            format_string = (
                '%(asctime)s | %(name)-30s | %(levelname)-8s | '
                '%(filename)s:%(lineno)d | %(message)s'
            )
        
        # Configura root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Remove handlers existentes
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Handler console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            format_string, datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root_logger.addHandler(console_handler)
        self._handlers.append(console_handler)
        
        # Handler arquivo
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                format_string, datefmt='%Y-%m-%d %H:%M:%S'
            ))
            root_logger.addHandler(file_handler)
            self._handlers.append(file_handler)
        
        # Configura níveis específicos por módulo
        if module_levels:
            for module, lvl in module_levels.items():
                logging.getLogger(module).setLevel(
                    getattr(logging, lvl.upper(), level)
                )
        
        # Silencia bibliotecas muito verbosas
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        
        logging.info(f"✅ Logging configurado - nível: {logging.getLevelName(level)}")
    
    def get_logger(self, name: str, level: Optional[int] = None) -> logging.Logger:
        """Retorna logger configurado."""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            if level:
                logger.setLevel(level)
            self._loggers[name] = logger
        return self._loggers[name]
    
    def add_file_handler(self, path: Union[str, Path], level: Optional[str] = None):
        """Adiciona handler de arquivo adicional."""
        handler = logging.FileHandler(path, encoding='utf-8')
        if level:
            handler.setLevel(getattr(logging, level.upper()))
        
        formatter = logging.Formatter(
            '%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(handler)
        self._handlers.append(handler)

# Singleton global
logger_manager = PyFoldsLogger()

def get_logger(name: str) -> logging.Logger:
    """Função conveniência para obter logger."""
    return logger_manager.get_logger(name)