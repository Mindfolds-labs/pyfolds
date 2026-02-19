# ADR-038 — Governança do `.fold/.mind`: auditoria de integridade e prompt operacional padronizado

**Nota de supersessão (2026-02-19):** este documento foi movido para legado e marcado como **Superseded**.
ADR canônico vigente: `docs/governance/adr/ADR-038-auditoria-fold-mind-governanca-execucao.md`.
Motivo: resolução de duplicidade de ADR-038, mantendo uma única referência oficial.

## Status
Superseded

## Contexto
Foi solicitada uma revisão objetiva do formato `.fold/.mind` para confirmar:
1. se o fluxo de serialização/desserialização está íntegro;
2. se há falhas lógicas aparentes na implementação atual;
3. se o time possui um prompt operacional reutilizável para execução via Codex com foco em validação técnica e governança.

Além disso, a demanda exige entrega completa no fluxo `ISSUE/EXEC + CSV + HUB`, com rastreabilidade documental e atualização de decisão arquitetural.

## Decisão
Adotar, para auditorias de formato `.fold/.mind`, o seguinte baseline obrigatório:

1. **Validação técnica mínima**
   - `PYTHONPATH=src pytest -q tests/unit/serialization/test_foldio.py`
   - `PYTHONPATH=src python -m py_compile src/pyfolds/serialization/foldio.py tests/unit/serialization/test_foldio.py`

2. **Hardening de assinatura digital (quando habilitada)**
   - Falhas de parsing/verificação de chave pública devem resultar em erro de segurança explícito (`FoldSecurityError`), não em exceções genéricas.

3. **Prompt operacional canônico para Codex**
   - manter um prompt pronto para executar auditoria de integridade `.fold/.mind`, detectar regressões e produzir evidências.

4. **Governança de entrega**
   - toda execução deve atualizar `ISSUE/EXEC`, `execution_queue.csv` e `HUB_CONTROLE.md` no mesmo ciclo.

## Alternativas consideradas
1. **Auditoria apenas textual (sem testes executados)**
   - Prós: menor custo.
   - Contras: baixo valor probatório.

2. **Auditoria com execução técnica + artefatos de governança (decisão adotada)**
   - Prós: evidência reproduzível, rastreabilidade e menor risco de regressão silenciosa.
   - Contras: maior esforço operacional.

## Consequências
- O formato `.fold/.mind` passa a ter um ritual de validação explícito e repetível.
- Falhas de verificação criptográfica ficam semanticamente claras para operação e auditoria.
- A execução segue alinhada ao fluxo de governança do projeto (ISSUE/EXEC/CSV/HUB).
