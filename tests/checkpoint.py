"""Mixin para checkpointing de modelos."""

import torch
import time
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class CheckpointMixin:
    """Mixin para adicionar funcionalidade de checkpoint a qualquer módulo."""
    
    def save(self, 
             path: Union[str, Path], 
             metrics: Optional[Dict[str, Any]] = None,
             metadata: Optional[Dict[str, Any]] = None,
             include_optimizer: bool = False) -> str:
        """
        Salva checkpoint completo do modelo.
        
        Args:
            path: Caminho para salvar (.pt ou .pth)
            metrics: Métricas adicionais para salvar
            metadata: Metadados extras
            include_optimizer: Se True, tenta salvar optimizer se existir
            
        Returns:
            Caminho absoluto do arquivo salvo
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'state_dict': self.state_dict(),
            'timestamp': time.time(),
            'version': getattr(self, '__version__', 'unknown'),
            'class': self.__class__.__name__
        }
        
        # Salva configuração se existir
        if hasattr(self, 'cfg'):
            checkpoint['config'] = self.cfg
            if hasattr(self.cfg, 'to_dict'):
                checkpoint['config_dict'] = self.cfg.to_dict()
        
        # Salva métricas
        if metrics:
            checkpoint['metrics'] = metrics
        
        # Salva metadados
        if metadata:
            checkpoint['metadata'] = metadata
        
        # Salva optimizer se existir e solicitado
        if include_optimizer and hasattr(self, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # Salva histórico se existir
        if hasattr(self, 'stats_acc') and hasattr(self.stats_acc, 'history'):
            checkpoint['history'] = self.stats_acc.history
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint salvo em: {path}")
        
        # Salva também um JSON com metadados para fácil leitura
        meta_path = path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'timestamp': checkpoint['timestamp'],
                'version': checkpoint['version'],
                'class': checkpoint['class'],
                'metrics': metrics or {},
                'metadata': metadata or {}
            }, f, indent=2)
        
        return str(path)
    
    @classmethod
    def load(cls, 
             path: Union[str, Path], 
             map_location: Optional[str] = None,
             **kwargs):
        """
        Carrega modelo a partir de checkpoint.
        
        Args:
            path: Caminho do checkpoint
            map_location: Device para carregar (cpu, cuda, etc)
            **kwargs: Argumentos extras para o construtor
            
        Returns:
            Instância do modelo carregada
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {path}")
        
        checkpoint = torch.load(path, map_location=map_location)
        
        # Se tem config no checkpoint, usa ela
        if 'config' in checkpoint:
            instance = cls(checkpoint['config'], **kwargs)
        elif 'config_dict' in checkpoint:
            from .config import MPJRDConfig
            config = MPJRDConfig.from_dict(checkpoint['config_dict'])
            instance = cls(config, **kwargs)
        else:
            instance = cls(**kwargs)
        
        # Carrega state dict
        instance.load_state_dict(checkpoint['state_dict'])
        
        # Carrega optimizer se existir
        if 'optimizer_state_dict' in checkpoint and hasattr(instance, 'optimizer'):
            instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Modelo carregado de: {path}")
        return instance
    
    def save_weights_only(self, path: Union[str, Path]) -> str:
        """Salva apenas os pesos (sem configuração)."""
        path = Path(path)
        torch.save(self.state_dict(), path)
        return str(path)
    
    def load_weights_only(self, path: Union[str, Path], strict: bool = True):
        """Carrega apenas os pesos."""
        path = Path(path)
        self.load_state_dict(torch.load(path), strict=strict)