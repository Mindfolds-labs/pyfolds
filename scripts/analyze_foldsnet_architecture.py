from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path


DATASETS: dict[str, tuple[tuple[int, int, int], int]] = {
    "mnist": ((1, 28, 28), 10),
    "cifar10": ((3, 32, 32), 10),
    "cifar100": ((3, 32, 32), 100),
}

MULTIPLIER = {"2L": 0.2, "4L": 1.0, "5L": 1.3, "6L": 1.7}


@dataclass
class FOLDSNetArchitectureStats:
    dataset: str
    variant: str
    input_shape: tuple[int, int, int]
    n_classes: int
    n_pixels: int
    n_retina: int
    n_lgn: int
    n_v1: int
    n_it: int
    classifier_weights: int
    classifier_bias: int
    classifier_params: int
    neuron_count_total: int
    approx_classifier_memory_mib_fp32: float


def _compute_stats(dataset: str, variant: str) -> FOLDSNetArchitectureStats:
    if dataset not in DATASETS:
        raise ValueError(f"dataset inválido: {dataset}")
    if variant not in MULTIPLIER:
        raise ValueError(f"variant inválido: {variant}")

    input_shape, n_classes = DATASETS[dataset]
    channels, height, width = input_shape
    n_pixels = channels * height * width

    n_retina_base = max(1, n_pixels // 16)
    n_retina = max(1, int(math.ceil(n_retina_base * MULTIPLIER[variant])))
    n_lgn = n_retina
    n_v1 = n_retina * 2
    n_it = n_retina

    classifier_weights = n_it * n_classes
    classifier_bias = n_classes
    classifier_params = classifier_weights + classifier_bias

    neuron_count_total = n_retina + n_lgn + n_v1 + n_it
    approx_classifier_memory_mib_fp32 = classifier_params * 4 / (1024**2)

    return FOLDSNetArchitectureStats(
        dataset=dataset,
        variant=variant,
        input_shape=input_shape,
        n_classes=n_classes,
        n_pixels=n_pixels,
        n_retina=n_retina,
        n_lgn=n_lgn,
        n_v1=n_v1,
        n_it=n_it,
        classifier_weights=classifier_weights,
        classifier_bias=classifier_bias,
        classifier_params=classifier_params,
        neuron_count_total=neuron_count_total,
        approx_classifier_memory_mib_fp32=round(approx_classifier_memory_mib_fp32, 6),
    )


def _render_markdown(rows: list[FOLDSNetArchitectureStats]) -> str:
    header = [
        "# FOLDSNet Architecture Report",
        "",
        "Este relatório resume topologia por camada e custo aproximado do classificador final (FP32).",
        "",
        "## Fluxo de camadas",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[Input] --> B[Retina]",
        "    B --> C[LGN]",
        "    C --> D[V1]",
        "    D --> E[IT]",
        "    E --> F[Linear Classifier]",
        "```",
        "",
        "## Tabela comparativa",
        "",
        "| dataset | variant | input_shape | classes | retina | lgn | v1 | it | total_neurons | classifier_params | classifier_mem_mib_fp32 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        header.append(
            f"| {row.dataset} | {row.variant} | {row.input_shape} | {row.n_classes} | {row.n_retina} | {row.n_lgn} | {row.n_v1} | {row.n_it} | {row.neuron_count_total} | {row.classifier_params} | {row.approx_classifier_memory_mib_fp32} |"
        )

    header.extend(
        [
            "",
            "## Notas",
            "",
            "- O número de neurônios em Retina/LGN/V1/IT escala com a variante (`2L < 4L < 5L < 6L`).",
            "- O custo de parâmetros explícitos aqui cobre principalmente o `nn.Linear` final.",
            "- A dinâmica bioinspirada principal está nos neurônios/camadas MPJRD usados dentro do FOLDSNet.",
            "- Mecanismos de EEG, engram e memória de consolidação são mais diretamente configuráveis no caminho MPJRD avançado, não como bloco dedicado no classificador FOLDSNet.",
        ]
    )

    return "\n".join(header) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analisa topologia/parametrização do FOLDSNet.")
    parser.add_argument("--datasets", nargs="+", default=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--variants", nargs="+", default=["2L", "4L", "5L", "6L"])
    parser.add_argument("--json-out", default="docs/assets/foldsnet_architecture_report.json")
    parser.add_argument("--md-out", default="docs/assets/FOLDSNET_ARCHITECTURE_REPORT.md")
    ns = parser.parse_args()

    rows = [_compute_stats(ds, var) for ds in ns.datasets for var in ns.variants]

    json_path = Path(ns.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")

    md_path = Path(ns.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_render_markdown(rows), encoding="utf-8")

    print(f"[OK] JSON: {json_path}")
    print(f"[OK] Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
