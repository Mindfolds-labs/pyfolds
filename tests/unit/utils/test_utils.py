#!/usr/bin/env python3
"""
TESTE ÚNICO PARA TODOS OS UTILITÁRIOS DO PyFolds

Este arquivo testa todas as funcionalidades do módulo utils:
- math.py (funções matemáticas)
- device.py (gerenciamento de device)
- types.py (tipos e enums)
- logging.py (logging profissional)

Execute com: pytest tests/test_utils.py -v
"""

import pytest
import torch
import logging
import re
import tempfile
import subprocess
import os
from pathlib import Path

# ===== IMPORTS DO PyFolds =====
from pyfolds.utils.logging import build_log_path, next_log_path, setup_run_logging

from pyfolds.utils import (
    # Math
    safe_div,
    clamp_rate,
    clamp_R,
    xavier_init,
    calculate_vc_dimension,
    
    # Device
    infer_device,
    ensure_device,
    get_device,
    
    # Types
    LearningMode,
    ConnectionType,
    ModeConfig,
    
    # Logging
    get_logger,
    PyFoldsLogger,
    TRACE_LEVEL,
    trace
)


# ============================================================================
# TESTES PARA MATH.PY
# ============================================================================

class TestMath:
    """Testes para funções matemáticas (math.py)."""
    
    def test_safe_div_basic(self):
        """Testa divisão normal."""
        x = torch.tensor([10.0, 20.0])
        y = torch.tensor([2.0, 5.0])
        result = safe_div(x, y)
        expected = torch.tensor([5.0, 4.0])
        assert torch.allclose(result, expected)
    
    def test_safe_div_by_zero(self):
        """Testa divisão por zero com epsilon."""
        x = torch.tensor([10.0])
        y = torch.tensor([0.0])
        result = safe_div(x, y, eps=1e-8)
        # 10 / (0 + 1e-8) ≈ 1e9
        assert result.item() > 1e8
    
    def test_clamp_rate_tensor(self):
        """Testa clamp_rate com tensor."""
        r = torch.tensor([-0.5, 0.3, 1.5, 2.0])
        clamped = clamp_rate(r)
        expected = torch.tensor([0.0, 0.3, 1.0, 1.0])
        assert torch.allclose(clamped, expected)
    
    def test_clamp_rate_float(self):
        """Testa clamp_rate com float."""
        assert clamp_rate(-0.5) == 0.0
        assert clamp_rate(0.3) == 0.3
        assert clamp_rate(1.5) == 1.0
        assert clamp_rate(2.0) == 1.0
    
    def test_clamp_R_tensor(self):
        """Testa clamp_R com tensor."""
        r = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        clamped = clamp_R(r)
        expected = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        assert torch.allclose(clamped, expected)
    
    def test_clamp_R_float(self):
        """Testa clamp_R com float."""
        assert clamp_R(-2.0) == -1.0
        assert clamp_R(-0.5) == -0.5
        assert clamp_R(0.0) == 0.0
        assert clamp_R(0.5) == 0.5
        assert clamp_R(2.0) == 1.0
    
    def test_xavier_init_shape(self):
        """Testa se xavier_init retorna shape correto."""
        shape = (10, 20)
        w = xavier_init(shape)
        assert w.shape == shape
    
    def test_xavier_init_gain(self):
        """Testa se gain altera escala."""
        w1 = xavier_init((1000, 1000), gain=1.0)
        w2 = xavier_init((1000, 1000), gain=2.0)
        assert w2.std() > w1.std()
    
    def test_calculate_vc_dimension(self):
        """Testa cálculo de VC-dimension."""
        vc = calculate_vc_dimension(100, 4, 32)
        assert vc > 0
        assert isinstance(vc, float)
        
        # Testa com diferentes parâmetros
        vc_small = calculate_vc_dimension(10, 2, 8)
        vc_large = calculate_vc_dimension(1000, 8, 64)
        assert vc_large > vc_small
        
        # Testa com avg_connections
        vc_dense = calculate_vc_dimension(100, 4, 32, avg_connections=1.0)
        vc_sparse = calculate_vc_dimension(100, 4, 32, avg_connections=0.1)
        assert vc_dense > vc_sparse


# ============================================================================
# TESTES PARA DEVICE.PY
# ============================================================================

class TestDevice:
    """Testes para gerenciamento de device (device.py)."""
    
    def test_infer_device_none(self):
        """Testa infer_device com None."""
        device = infer_device(None)
        assert device.type == 'cpu'
    
    def test_infer_device_tensor_cpu(self):
        """Testa infer_device com tensor CPU."""
        x = torch.randn(10, device='cpu')
        device = infer_device(x)
        assert device.type == 'cpu'
    
    def test_infer_device_tensor_cuda(self):
        """Testa infer_device com tensor CUDA (se disponível)."""
        if torch.cuda.is_available():
            x = torch.randn(10, device='cuda')
            device = infer_device(x)
            assert device.type == 'cuda'
    
    def test_infer_device_dict(self):
        """Testa infer_device com dicionário."""
        d = {
            'a': torch.randn(10, device='cpu'),
            'b': torch.randn(10)
        }
        device = infer_device(d)
        assert device.type == 'cpu'
    
    def test_infer_device_empty_dict(self):
        """Testa infer_device com dicionário vazio."""
        device = infer_device({})
        assert device.type == 'cpu'
    
    def test_ensure_device_no_change(self):
        """Testa ensure_device sem device especificado."""
        x = torch.randn(10, device='cpu')
        y = ensure_device(x)
        assert y.device.type == 'cpu'
        # Deve ser o mesmo tensor (ou cópia?)
        assert torch.allclose(x, y)
    
    def test_ensure_device_move(self):
        """Testa ensure_device movendo tensor."""
        x = torch.randn(10, device='cpu')
        
        # Se CUDA disponível, testa movimento CPU->CUDA
        if torch.cuda.is_available():
            y = ensure_device(x, torch.device('cuda'))
            assert y.device.type == 'cuda'
            assert torch.allclose(x.cpu(), y.cpu())
    
    def test_get_device_default(self):
        """Testa get_device sem argumentos."""
        device = get_device()
        if torch.cuda.is_available():
            assert device.type == 'cuda'
        else:
            assert device.type == 'cpu'
    
    def test_get_device_specific(self):
        """Testa get_device com device específico."""
        device = get_device('cpu')
        assert device.type == 'cpu'
        
        if torch.cuda.is_available():
            device = get_device('cuda')
            assert device.type == 'cuda'


# ============================================================================
# TESTES PARA TYPES.PY
# ============================================================================

class TestTypes:
    """Testes para tipos e enums (types.py)."""
    
    def test_learning_mode_values(self):
        """Testa valores do enum LearningMode."""
        assert LearningMode.ONLINE.value == "online"
        assert LearningMode.BATCH.value == "batch"
        assert LearningMode.SLEEP.value == "sleep"
        assert LearningMode.INFERENCE.value == "inference"
    
    def test_learning_mode_description(self):
        """Testa descriptions do LearningMode."""
        assert "vigília" in LearningMode.ONLINE.description
        assert "lote" in LearningMode.BATCH.description
        assert "Sono" in LearningMode.SLEEP.description
        assert "produção" in LearningMode.INFERENCE.description
    
    def test_learning_rate_multiplier(self):
        """Testa multiplicadores de learning rate."""
        assert LearningMode.ONLINE.learning_rate_multiplier == 5.0
        assert LearningMode.BATCH.learning_rate_multiplier == 0.2
        assert LearningMode.SLEEP.learning_rate_multiplier == 0.0
        assert LearningMode.INFERENCE.learning_rate_multiplier == 0.0
    
    def test_is_learning(self):
        """Testa is_learning()."""
        assert LearningMode.ONLINE.is_learning() is True
        assert LearningMode.BATCH.is_learning() is True
        assert LearningMode.SLEEP.is_learning() is False
        assert LearningMode.INFERENCE.is_learning() is False
    
    def test_is_consolidating(self):
        """Testa is_consolidating()."""
        assert LearningMode.ONLINE.is_consolidating() is False
        assert LearningMode.BATCH.is_consolidating() is False
        assert LearningMode.SLEEP.is_consolidating() is True
        assert LearningMode.INFERENCE.is_consolidating() is False
    
    def test_connection_type(self):
        """Testa ConnectionType enum."""
        assert ConnectionType.DENSE.value == "dense"
        assert ConnectionType.SPARSE.value == "sparse"
        assert ConnectionType.ENTANGLED.value == "entangled"
    
    def test_mode_config_defaults(self):
        """Testa valores padrão do ModeConfig."""
        config = ModeConfig()
        assert config.online_learning_rate_mult == 5.0
        assert config.batch_learning_rate_mult == 0.2
        assert config.sleep_consolidation_factor == 0.1
    
    def test_mode_config_get_learning_rate(self):
        """Testa get_learning_rate()."""
        config = ModeConfig()
        base_lr = 0.01
        
        lr_online = config.get_learning_rate(base_lr, LearningMode.ONLINE)
        assert lr_online == base_lr * 5.0
        
        lr_batch = config.get_learning_rate(base_lr, LearningMode.BATCH)
        assert lr_batch == base_lr * 0.2
        
        lr_sleep = config.get_learning_rate(base_lr, LearningMode.SLEEP)
        assert lr_sleep == 0.0
        
        lr_inf = config.get_learning_rate(base_lr, LearningMode.INFERENCE)
        assert lr_inf == 0.0
    
    def test_mode_config_get_consolidation_factor(self):
        """Testa get_consolidation_factor()."""
        config = ModeConfig()
        
        assert config.get_consolidation_factor(LearningMode.ONLINE) == 0.0
        assert config.get_consolidation_factor(LearningMode.BATCH) == 0.0
        assert config.get_consolidation_factor(LearningMode.SLEEP) == 0.1
        assert config.get_consolidation_factor(LearningMode.INFERENCE) == 0.0


# ============================================================================
# TESTES PARA LOGGING.PY
# ============================================================================

class TestLogging:
    """Testes para logging profissional (logging.py)."""
    
    def test_trace_level_defined(self):
        """Testa se TRACE_LEVEL está definido."""
        assert TRACE_LEVEL == 5
        assert hasattr(logging, 'TRACE')
    
    def test_trace_method_exists(self):
        """Testa se método trace foi adicionado ao Logger."""
        logger = logging.getLogger('test')
        assert hasattr(logger, 'trace')
        assert callable(logger.trace)
    
    def test_get_logger(self):
        """Testa get_logger()."""
        logger = get_logger('test.logger')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test.logger'
    
    def test_pyfolds_logger_singleton(self):
        """Testa se PyFoldsLogger é singleton."""
        logger1 = PyFoldsLogger()
        logger2 = PyFoldsLogger()
        assert logger1 is logger2
    
    def test_logger_setup_basic(self):
        """Testa configuração básica do logger com console silencioso por padrão."""
        import inspect

        signature = inspect.signature(PyFoldsLogger.setup)
        assert signature.parameters["console"].default is False

        logger_manager = PyFoldsLogger()
        logger_manager.setup(level="INFO")

        logger = get_logger('test.setup')
        logger.info("Mensagem de teste")
    
    def test_logger_trace_output(self):
        """Testa se nível TRACE pode ser usado sem erro."""
        logger_manager = PyFoldsLogger()
        logger_manager.setup(level="TRACE")

        logger = get_logger('test.trace')
        logger.trace("Mensagem TRACE")

        assert logger.isEnabledFor(TRACE_LEVEL)
    
    def test_logger_trace_silent_when_lower_level(self, capsys):
        """Testa se trace não loga quando nível mais baixo."""
        logger_manager = PyFoldsLogger()
        logger_manager.setup(level="INFO")
        
        logger = get_logger('test.trace.silent')
        logger.trace("Esta mensagem NÃO deve aparecer")
        
        captured = capsys.readouterr()
        assert "Esta mensagem NÃO deve aparecer" not in captured.out
    

    def test_import_pyfolds_does_not_add_stream_handler(self):
        """Import não deve configurar handlers automaticamente."""
        cmd = [
            "python",
            "-c",
            (
                "import logging; "
                "before=len(logging.getLogger().handlers); "
                "import pyfolds; "
                "after=len(logging.getLogger().handlers); "
                "print(f'{before}:{after}')"
            ),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        before, after = map(int, proc.stdout.strip().split(":"))
        assert before == after

    def test_setup_run_logging_returns_logger_and_path(self, tmp_path):
        """Helper de setup deve criar arquivo incremental e logger configurado."""
        logger, log_path = setup_run_logging(
            app="pyfolds",
            version="2.0.0",
            log_dir=tmp_path,
            console=False,
            fixed_layout=True,
        )
        logger.info("linha setup_run")

        assert log_path.exists()
        assert log_path.name.startswith("001_pyfolds_2.0.0_")
        assert "linha setup_run" in log_path.read_text()

    def test_logger_file_handler(self, tmp_path):
        """Testa se logger escreve em arquivo."""
        log_file = tmp_path / "test.log"
        
        logger_manager = PyFoldsLogger()
        logger_manager.setup(level="DEBUG", log_file=log_file)
        
        logger = get_logger('test.file')
        logger.debug("Mensagem para arquivo")
        
        # Força flush
        for handler in logger.handlers:
            handler.flush()
        
        assert log_file.exists()
        content = log_file.read_text()
        assert "Mensagem para arquivo" in content
    

    def test_logger_console_false_keeps_terminal_clean(self, tmp_path, capsys):
        """Com console=False, logs não devem ir para stdout/stderr."""
        log_file = tmp_path / "silent.log"

        logger_manager = PyFoldsLogger()
        logger_manager.setup(level="INFO", log_file=log_file, console=False)

        logger = get_logger('test.silent')
        logger.info("arquivo-only")

        captured = capsys.readouterr()
        assert "arquivo-only" not in captured.out
        assert "arquivo-only" not in captured.err
        assert "arquivo-only" in log_file.read_text()

    def test_logger_fixed_layout(self, tmp_path):
        """Valida layout fixo de colunas para auditoria."""
        log_file = tmp_path / "cobol.log"

        logger_manager = PyFoldsLogger()
        logger_manager.setup(level="INFO", log_file=log_file, console=False, fixed_layout=True)

        logger = get_logger('test.fixed')
        logger.info("mensagem\ncom quebra")

        line = log_file.read_text().strip().splitlines()[-1]
        pattern = re.compile(r"^\d{8} \d{6}\.\d{3} \| [A-Z]{4,8}\s* \| .{35} \| .{14} \| .+$")
        assert pattern.match(line)
        assert "\\n" in line

    def test_build_log_path_increments_counter(self, tmp_path):
        """Valida nome incremental NNN_app_version_timestamp.log."""
        p1 = build_log_path(tmp_path, app="pyfolds", version="2.0.0")
        p1.write_text("x")
        p2 = build_log_path(tmp_path, app="pyfolds", version="2.0.0")

        assert p1.name.startswith("001_pyfolds_2.0.0_")
        assert p2.name.startswith("002_pyfolds_2.0.0_")


    def test_next_log_path_alias(self, tmp_path):
        """`next_log_path` deve manter contrato de nome incremental."""
        p1 = next_log_path(tmp_path, app="pyfolds", version="2.0.0")
        p1.write_text("x")
        p2 = next_log_path(tmp_path, app="pyfolds", version="2.0.0")

        assert p1.name.startswith("001_pyfolds_2.0.0_")
        assert p2.name.startswith("002_pyfolds_2.0.0_")

    def test_logger_circular_buffer_file_handler(self, tmp_path):
        """Testa escrita em TXT com comportamento circular por número de linhas."""
        log_file = tmp_path / "circular.log"

        logger_manager = PyFoldsLogger()
        logger_manager.setup(level="INFO", log_file=log_file, circular_buffer_lines=3)

        logger = get_logger('test.circular')
        for idx in range(5):
            logger.info(f"linha-{idx}")

        content = log_file.read_text().strip().splitlines()
        assert len(content) == 3
        assert any("linha-2" in line for line in content)
        assert any("linha-3" in line for line in content)
        assert any("linha-4" in line for line in content)

    def test_multiple_loggers_same_name(self):
        """Testa se get_logger com mesmo nome retorna mesmo logger."""
        logger1 = get_logger('test.same')
        logger2 = get_logger('test.same')
        assert logger1 is logger2
    
    def test_module_levels(self):
        """Testa configuração de níveis por módulo."""
        logger_manager = PyFoldsLogger()
        logger_manager.setup(
            level="INFO",
            module_levels={
                'test.module': 'DEBUG',
                'test.other': 'WARNING'
            }
        )
        
        test_logger = logging.getLogger('test.module')
        other_logger = logging.getLogger('test.other')
        
        assert test_logger.level == logging.DEBUG
        assert other_logger.level == logging.WARNING


# ============================================================================
# TESTES DE INTEGRAÇÃO (usando múltiplos módulos)
# ============================================================================

class TestIntegration:
    """Testes que integram múltiplos módulos."""
    
    def test_logging_with_device(self):
        """Testa logging + device juntos."""
        logger = get_logger('test.integration')
        device = get_device()
        
        logger.info(f"Device detectado: {device}")
        # Não podemos testar a saída facilmente, mas pelo menos não quebra
    
    def test_learning_mode_with_math(self):
        """Testa LearningMode + funções matemáticas."""
        mode = LearningMode.ONLINE
        rate = clamp_rate(1.5)  # Deveria ser 1.0
        
        # Em modo ONLINE, learning rate é 5x
        # Isso é apenas um teste conceitual
        assert rate == 1.0
        assert mode.learning_rate_multiplier == 5.0
    
    def test_device_with_tensor_ops(self):
        """Testa device com operações de tensor."""
        device = get_device('cpu')
        x = torch.randn(10, device=device)
        y = torch.randn(10, device=device)
        
        z = safe_div(x, y)
        assert z.device.type == 'cpu'


# ============================================================================
# MAIN (para executar diretamente)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTANDO TODOS OS UTILITÁRIOS DO PyFolds")
    print("=" * 60)
    
    # Configura logging para os testes
    PyFoldsLogger().setup(level="INFO")
    
    # Roda testes manualmente (ou usa pytest)
    pytest.main([__file__, "-v"])