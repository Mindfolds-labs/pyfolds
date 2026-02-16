"""Catálogo simples de soluções de governança do projeto Fold Mind.

Este arquivo funciona como referência programática para iniciativas em aberto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Solucao:
    area: str
    descricao: str
    status: str


def catalogo_solucoes() -> Dict[str, List[Solucao]]:
    """Retorna catálogo agrupado por macroárea."""

    return {
        "qualidade": [
            Solucao(
                area="testes",
                descricao="Adicionar regressão automatizada para bugs críticos confirmados.",
                status="planejado",
            ),
            Solucao(
                area="telemetria",
                descricao="Padronizar eventos para facilitar análise temporal de falhas.",
                status="em_andamento",
            ),
        ],
        "documentacao": [
            Solucao(
                area="governanca",
                descricao="Manter pacote de documentos canônicos vinculado no README.",
                status="concluido",
            ),
            Solucao(
                area="onboarding",
                descricao="Criar trilha guiada de leitura para novos colaboradores.",
                status="planejado",
            ),
        ],
    }


if __name__ == "__main__":
    for macroarea, solucoes in catalogo_solucoes().items():
        print(f"[{macroarea}]")
        for item in solucoes:
            print(f"- {item.area}: {item.descricao} ({item.status})")
