# EXECUÇÃO: Adicionar dependência linkify-it-py

## Tarefa
Adicionar `linkify-it-py>=2.0` ao arquivo `requirements-docs.txt` para corrigir erro de build da documentação.

## Contexto
Issue: ISSUE-001-docs-dependency-linkify
Erro: `ModuleNotFoundError: Linkify enabled but not installed`

## Passos para Execução

1. **Adicionar dependência:**
   ```bash
   echo "linkify-it-py>=2.0" >> requirements-docs.txt
   ```
2. **Instalar dependências de documentação:**
   ```bash
   pip install -r requirements-docs.txt
   ```
3. **Validar build local da documentação:**
   ```bash
   sphinx-build -b html docs docs/_build/html
   ```

## Riscos e mitigação
- Versão incompatível do `linkify-it-py` com MyST Parser | Testar localmente antes do commit.
- Quebra em outras dependências transitivas | Usar range de versão flexível (`>=2.0,<3.0`) se necessário.

---
**Contexto Técnico:**
O erro `ModuleNotFoundError: Linkify enabled but not installed` ocorre porque o MyST Parser tem a extensão `linkify` habilitada, mas o pacote `linkify-it-py` não está listado nas dependências de documentação.
