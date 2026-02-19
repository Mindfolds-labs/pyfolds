#!/usr/bin/env python3
"""Extrai violações críticas/invariantes dos logs do Sheer Audit e atualiza failure_register.csv."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DEFAULT_REPORTS_DIR = Path("docs/sheer-audit/reports")
DEFAULT_REGISTER_PATH = Path("docs/development/failure_register.csv")

CSV_FIELDS = [
    "id",
    "tipo",
    "descricao",
    "impacto",
    "status",
    "issue_correcao",
    "assinatura_erro",
    "arquivo_afetado",
    "caminho_log",
    "data_registro",
    "ultima_ocorrencia",
]

KEYWORD_PATTERNS = {
    "critical": re.compile(r"\b(critical|cr[ií]tic[oa]|fatal|high severity|severe)\b", re.IGNORECASE),
    "invariant": re.compile(r"\b(invariant|invariante)\b", re.IGNORECASE),
    "violation": re.compile(r"\b(violat(?:ion|e[ds]?)|viola[cç][aã]o|breach|quebra)\b", re.IGNORECASE),
    "assertion": re.compile(r"\b(assert(?:ion)?(?:error)?|failed|failure)\b", re.IGNORECASE),
    "file_hint": re.compile(r"\b([\w./-]+\.(?:py|md|txt|json|yaml|yml|toml))\b"),
}


@dataclass(slots=True)
class Finding:
    tipo: str
    descricao: str
    impacto: str
    arquivo_afetado: str
    caminho_log: str
    assinatura_erro: str


def iter_report_files(reports_dir: Path) -> Iterable[Path]:
    for path in sorted(reports_dir.glob("**/*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".log", ".out"}:
            yield path


def summarize_line(line: str) -> str:
    normalized = re.sub(r"\s+", " ", line).strip(" -:*\t")
    return normalized[:200] if len(normalized) > 200 else normalized


def classify_tipo(line: str) -> str:
    if KEYWORD_PATTERNS["invariant"].search(line):
        return "violacao_invariante"
    if KEYWORD_PATTERNS["critical"].search(line):
        return "violacao_critica"
    return "violacao"


def classify_severity(tipo: str, line: str) -> str:
    if tipo == "violacao_invariante":
        return "Alto"
    high_cues = [
        KEYWORD_PATTERNS["critical"],
        re.compile(r"\b(segfault|panic|corrupt(?:ion)?|security|unsafe)\b", re.IGNORECASE),
    ]
    if any(pattern.search(line) for pattern in high_cues):
        return "Alto"
    return "Médio"


def infer_impacto(severity: str, tipo: str) -> str:
    if severity == "Alto" and tipo == "violacao_invariante":
        return "Alto - quebra de invariante funcional"
    if severity == "Alto":
        return "Alto - falha crítica potencial"
    return "Médio - requer análise e mitigação"


def infer_affected_file(line: str, log_path: Path) -> str:
    found = KEYWORD_PATTERNS["file_hint"].findall(line)
    if not found:
        return "N/A"
    for candidate in found:
        if candidate != log_path.name:
            return candidate
    return found[0]


def extract_findings_from_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        has_core_signal = (
            KEYWORD_PATTERNS["violation"].search(line)
            or KEYWORD_PATTERNS["invariant"].search(line)
            or (KEYWORD_PATTERNS["critical"].search(line) and KEYWORD_PATTERNS["assertion"].search(line))
        )
        if not has_core_signal:
            continue

        tipo = classify_tipo(line)
        severity = classify_severity(tipo, line)
        description = summarize_line(line)
        signature_base = f"{path.as_posix()}::{description.lower()}::{tipo}"
        assinatura = hashlib.sha1(signature_base.encode("utf-8")).hexdigest()[:16]

        findings.append(
            Finding(
                tipo=tipo,
                descricao=description,
                impacto=infer_impacto(severity, tipo),
                arquivo_afetado=infer_affected_file(line, path),
                caminho_log=path.as_posix(),
                assinatura_erro=assinatura,
            )
        )
    return findings


def ensure_register(register_path: Path) -> None:
    register_path.parent.mkdir(parents=True, exist_ok=True)
    if register_path.exists() and register_path.stat().st_size > 0:
        return
    with register_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()


def load_register(register_path: Path) -> list[dict[str, str]]:
    ensure_register(register_path)
    with register_path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def next_id(existing_rows: list[dict[str, str]]) -> str:
    max_n = 0
    for row in existing_rows:
        match = re.fullmatch(r"FAIL-(\d{4,})", row.get("id", ""))
        if match:
            max_n = max(max_n, int(match.group(1)))
    return f"FAIL-{max_n + 1:04d}"


def upsert_findings(register_rows: list[dict[str, str]], findings: list[Finding], now_iso: str) -> tuple[list[dict[str, str]], int, int]:
    by_signature = {row.get("assinatura_erro", ""): row for row in register_rows}
    inserted = 0
    updated = 0

    for finding in findings:
        current = by_signature.get(finding.assinatura_erro)
        if current is not None:
            current["ultima_ocorrencia"] = now_iso
            current["caminho_log"] = finding.caminho_log
            updated += 1
            continue

        new_id = next_id(register_rows)
        row = {
            "id": new_id,
            "tipo": finding.tipo,
            "descricao": finding.descricao,
            "impacto": finding.impacto,
            "status": "aberto",
            "issue_correcao": "",
            "assinatura_erro": finding.assinatura_erro,
            "arquivo_afetado": finding.arquivo_afetado,
            "caminho_log": finding.caminho_log,
            "data_registro": now_iso,
            "ultima_ocorrencia": now_iso,
        }
        register_rows.append(row)
        by_signature[finding.assinatura_erro] = row
        inserted += 1

    return register_rows, inserted, updated


def write_register(register_path: Path, rows: list[dict[str, str]]) -> None:
    with register_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--register", type=Path, default=DEFAULT_REGISTER_PATH)
    args = parser.parse_args()

    if not args.reports_dir.exists():
        raise SystemExit(f"Diretório de relatórios não encontrado: {args.reports_dir}")

    all_findings: list[Finding] = []
    for report_file in iter_report_files(args.reports_dir):
        all_findings.extend(extract_findings_from_file(report_file))

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = load_register(args.register)
    rows, inserted, updated = upsert_findings(rows, all_findings, now_iso)
    if inserted or updated:
        write_register(args.register, rows)

    print(
        f"Relatórios analisados: {len(list(iter_report_files(args.reports_dir)))} | "
        f"achados: {len(all_findings)} | novos: {inserted} | atualizados: {updated}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
