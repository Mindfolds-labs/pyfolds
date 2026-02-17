# Estratégia de Testes

Este documento descreve a estratégia de testes para manter estabilidade funcional e científica no PyFolds.

## 1) Objetivos
- Garantir corretude matemática das rotinas principais.
- Evitar regressões em mudanças incrementais.
- Preservar comportamento esperado entre versões.

## 2) Pirâmide de testes

### Unitários
Cobrem funções e classes isoladas, com foco em:
- validação de entradas e erros,
- invariantes numéricos,
- contratos de API pública.

### Integração
Cobrem fluxo entre módulos:
- forward + plasticidade,
- serialização + desserialização,
- monitoramento + validações internas.

### Regressão
Cobrem estabilidade histórica:
- benchmarks sintéticos,
- comparações com saídas de referência,
- cenários críticos já incidentados.

## 3) Cobertura mínima
- Meta recomendada: **80%** para módulos críticos.
- Prioridade de cobertura: `core`, `serialization`, `monitoring`.

## 4) Boas práticas por PR
1. Adicionar teste para cada bug corrigido.
2. Evitar testes frágeis dependentes de timing instável.
3. Isolar fixtures e dados de teste reutilizáveis.
4. Nomear testes com padrão `test_<comportamento>_<contexto>`.

## 5) Comandos locais
```bash
PYTHONPATH=src pytest tests/ -v
PYTHONPATH=src pytest tests/ -q -k "core or serialization"
python -m compileall src/
```

## 6) Critérios para aprovação
- Testes novos e antigos devem passar no CI.
- Não introduzir redução relevante de cobertura em módulos críticos.
- Alterações de comportamento devem atualizar docs e testes.

## 7) Cenários mínimos por domínio
- **Core neuronal:** soma, dendritos, sinapses, limites e broadcasting.
- **Serialização:** round-trip, leitura parcial e erros de integridade.
- **API de alto nível:** parâmetros padrão, tipos e mensagens de erro.
- **Governança de docs:** links internos e consistência de estrutura.

## 8) Falhas comuns a evitar
- Ajustar implementação sem ajustar testes esperados.
- Cobrir apenas caso feliz e ignorar entradas inválidas.
- Deixar testes acoplados a detalhes internos não contratuais.

## 9) Evolução contínua
A estratégia deve ser revisada a cada release quando houver:
- novos módulos em `src/`,
- mudanças de formato de arquivo,
- atualização de políticas de qualidade.
