# Prompt técnico (engenharia sênior) — Runner robusto para PyFolds

Implemente um runner genérico para executar scripts Python de treino/debug com captura total de erros e evidências.

## Objetivo
- Capturar stdout/stderr de qualquer script Python (`SyntaxError`, `ImportError`, `RuntimeError`, etc.).
- Salvar log estruturado em arquivo incremental.
- Retornar exit code correto para PowerShell/CI.
- Permitir `tee` opcional para progresso em terminal.

## Entregável obrigatório
- Arquivo `run_pyfolds.py` com:
  - `run_script(script_path: str, args: list[str], tee: bool = True) -> int`
  - CLI: `python run_pyfolds.py script.py [args...] [--no-tee] [--log-dir logs] [--app-name pyfolds]`

## Requisitos técnicos
1. Use `subprocess.Popen` com `stdout=PIPE`, `stderr=PIPE`, `text=True`, `bufsize=1`.
2. Use threads + queue para consumir stdout/stderr sem deadlock.
3. Prefixe linhas no log com `[STDOUT]` e `[STDERR]`.
4. Header obrigatório:
   - `PYFOLDS EXECUTION RUN`
   - versão detectada do pacote
   - script/command
   - timestamp de início
5. Footer obrigatório:
   - `EXIT CODE`
   - `DURATION`
   - timestamp de fim
6. Nome de log: `logs/NNN_pyfolds_VERSION_YYYYMMDD_HHMMSS.log`.
7. Finalize com `sys.exit(exit_code)`.

## Critérios de aceite
- Erro de sintaxe vai para log.
- Runtime error vai para log.
- Exit code do script é propagado.
- `tee=True` mostra progresso no terminal.
- `tee=False` mantém terminal silencioso.

## Extensões futuras sugeridas
- rotação de logs
- monitoramento GPU
- watchdog de travamento
- restart automático com política de retry
- checkpoint recovery
- export de métricas CSV


## Implementado (baseline robusto)
- watchdog por timeout (`--timeout-seconds`)
- retry automático (`--max-retries`)
- métricas CSV por execução (`--metrics-csv`)
- monitoramento GPU best-effort via `nvidia-smi` (`--monitor-gpu`)
- resume por checkpoint em retry (`--checkpoint-path`, `--resume-arg`)
- rotação simples de logs (retenção dos 200 mais novos)
