#!/usr/bin/env python3
"""Valida política de regressão por falha no failure_register.csv.

Regras:
1) Falhas encerradas/resolvidas só podem ser fechadas com teste de regressão associado.
2) Evidência de regressão deve referenciar explicitamente o ID FAIL-XXXX da linha.
3) Status de cobertura deve seguir vocabulário controlado.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

DEFAULT_REGISTER = Path("docs/development/failure_register.csv")

REQUIRED_COLUMNS = [
    "id",
    "status",
    "status_cobertura",
    "teste_regressao",
    "evidencia_regressao",
]

STATUS_COBERTURA_VALIDO = {"aberta", "em_correcao", "coberta", "validada"}
STATUS_FECHAMENTO = {"fechado", "encerrado", "resolvido", "concluido", "concluida", "done", "closed"}
FAIL_ID_RE = re.compile(r"FAIL-\d{3,}", re.IGNORECASE)


def norm(value: str | None) -> str:
    return "" if value is None else value.strip()


def is_closing_status(value: str) -> bool:
    lowered = norm(value).lower()
    return any(token in lowered for token in STATUS_FECHAMENTO)


def has_fail_reference(text: str, fail_id: str) -> bool:
    if not text:
        return False
    return fail_id.lower() in text.lower()


def validate(register_path: Path) -> list[str]:
    with register_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            return [
                "failure_register.csv sem colunas obrigatórias para política de regressão: "
                + ", ".join(missing)
            ]

        errors: list[str] = []
        for line_no, row in enumerate(reader, start=2):
            fail_id = norm(row.get("id"))
            if not fail_id:
                continue
            if not FAIL_ID_RE.fullmatch(fail_id):
                errors.append(f"L{line_no}: ID inválido '{fail_id}' (esperado FAIL-XXX).")

            status_cobertura = norm(row.get("status_cobertura")).lower()
            if status_cobertura not in STATUS_COBERTURA_VALIDO:
                errors.append(
                    f"L{line_no}: status_cobertura inválido '{status_cobertura or '-'}' "
                    f"(use: {', '.join(sorted(STATUS_COBERTURA_VALIDO))})."
                )

            teste = norm(row.get("teste_regressao"))
            evidencia = norm(row.get("evidencia_regressao"))
            status = norm(row.get("status"))

            fechamento = is_closing_status(status)
            coberta_ou_validada = status_cobertura in {"coberta", "validada"}

            if fechamento or coberta_ou_validada:
                if not teste:
                    errors.append(
                        f"L{line_no}: {fail_id} não pode ser fechada/coberta sem teste_regressao preenchido."
                    )
                if not evidencia:
                    errors.append(
                        f"L{line_no}: {fail_id} não pode ser fechada/coberta sem evidencia_regressao preenchida."
                    )

            if teste and not has_fail_reference(teste, fail_id):
                errors.append(
                    f"L{line_no}: teste_regressao deve referenciar o ID da falha ({fail_id})."
                )
            if evidencia and not has_fail_reference(evidencia, fail_id):
                errors.append(
                    f"L{line_no}: evidencia_regressao deve referenciar o ID da falha ({fail_id})."
                )

        return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path, default=DEFAULT_REGISTER)
    args = parser.parse_args()

    if not args.register.exists():
        print(f"Arquivo não encontrado: {args.register}", file=sys.stderr)
        return 2

    errors = validate(args.register)
    if errors:
        print("Falha na política de regressão por falha:", file=sys.stderr)
        for msg in errors:
            print(f"- {msg}", file=sys.stderr)
        return 1

    print("Política de regressão validada com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
