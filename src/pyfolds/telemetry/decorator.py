"""Decorador para telemetria automática."""

import time
import functools
from typing import Optional, Callable, Any
from .controller import TelemetryController
from .events import forward_event, commit_event, sleep_event


def telemetry(
    phase: str = "forward",
    sample_rate: Optional[float] = None,
    capture_args: bool = False,
    capture_return: bool = False
):
    """
    Decorador para coletar telemetria automaticamente.
    
    Args:
        phase: Fase do evento ('forward', 'commit', 'sleep')
        sample_rate: Taxa de amostragem (ex: 0.1 = 10% das chamadas)
        capture_args: Se True, captura argumentos da função
        capture_return: Se True, captura valor de retorno
    
    Example:
        >>> @telemetry(phase="forward", capture_return=True)
        ... def forward_pass(self, x):
        ...     # ... código ...
        ...     return {'spike_rate': 0.15, 'theta': 4.5}
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Verifica se tem telemetria
            if not hasattr(self, 'telemetry') or not self.telemetry.enabled():
                return func(self, *args, **kwargs)
            
            telem: TelemetryController = self.telemetry
            
            # Amostragem probabilística
            if sample_rate is not None and not telem.should_emit_sample(sample_rate):
                return func(self, *args, **kwargs)
            
            # Executa função
            start_time = time.perf_counter()
            result = func(self, *args, **kwargs)
            duration = time.perf_counter() - start_time
            
            # ===== CORREÇÃO: Prioriza step_id do objeto =====
            # Tenta pegar step_id do próprio objeto, se não existir usa do controller
            step_id = getattr(self, 'step_id', telem.step_count)
            mode = getattr(self, 'mode', 'unknown')
            neuron_id = getattr(self, 'neuron_id', None)
            
            # Coleta telemetria (se deve emitir baseado no step_id correto)
            if telem.should_emit(step_id):
                payload = {}
                
                if capture_args:
                    payload['args'] = str(args)
                    payload['kwargs'] = str(kwargs)
                
                if capture_return and isinstance(result, dict):
                    payload.update(result)
                
                payload['duration_ms'] = duration * 1000
                
                # Determina função de evento
                if phase == "forward":
                    event_fn = forward_event
                elif phase == "commit":
                    event_fn = commit_event
                else:
                    event_fn = sleep_event
                
                telem.emit(event_fn(
                    step_id=step_id,  # ✅ Agora usa o step_id correto!
                    mode=mode,
                    neuron_id=neuron_id,
                    **payload
                ))
            
            # Incrementa step_count do controller (mantém para fallback)
            telem.step_count += 1
            return result
        
        return wrapper
    return decorator