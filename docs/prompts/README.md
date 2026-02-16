# Prompts Operacionais – PyFolds

Esta pasta contém prompts estruturados para execução técnica assistida
via ChatGPT, Codex ou ferramentas similares.

O objetivo é padronizar melhorias no projeto, garantindo:

- Mudanças pequenas e seguras  
- Rastreabilidade completa (Issue → Fila → PR)  
- Governança técnica consistente  
- Testes obrigatórios  
- Documentação alinhada  

---

## 🎯 Filosofia

O sistema de prompts existe para transformar ideias em execução organizada.

Fluxo padrão:

Ideia → Prompt → Issue → Registro na Fila (CSV) → Implementação → Testes → PR → Merge

Nenhuma melhoria deve ser feita fora desse fluxo.

---

## 📂 Estrutura da Pasta

- `PROMPT_GERAL.md`  
  Prompt principal para qualquer melhoria incremental.

- `PROMPT_TESTES.md`  
  Focado em cobertura e qualidade de testes.

- `PROMPT_SERIALIZACAO.md`  
  Melhorias relacionadas ao formato `.fold` / `.mind`.

- `PROMPT_BENCHMARK.md`  
  Performance, latência e memória.

- `PROMPT_API.md`  
  Revisão e estabilidade da API pública.

- `PROMPT_ROADMAP.md`  
  Planejamento técnico em ciclos de sprint.

- `PROMPT_AUDITORIA.md`  
  Auditoria técnica completa do projeto.

---

## 🔒 Regras de Execução

Sempre que utilizar um prompt:

1. Trabalhar em branch dedicada.
2. Criar ou atualizar Issue correspondente.
3. Registrar na fila em:
   - `docs/development/execution_queue.csv`
4. Sincronizar o HUB:
   - `python tools/sync_hub.py`
5. Executar testes (`pytest`).
6. Atualizar `CHANGELOG.md` apenas se houver impacto externo.

---

## ⚠️ Restrições Importantes

- Nunca quebrar API pública sem ADR.
- Nunca alterar formato `.fold` sem decisão formal.
- Nunca fazer refactor grande sem cobertura de testes.
- Nunca misturar múltiplas melhorias na mesma PR.

---

## 🧠 Como Usar

1. Abra o arquivo `PROMPT_GERAL.md`.
2. Cole o conteúdo no Codex/ChatGPT.
3. No final do prompt, descreva o objetivo da melhoria.
4. Revise a saída.
5. Execute o PR conforme checklist.

---

## 📌 Nota Final

Esta pasta não é documentação de usuário.

É ferramenta operacional de engenharia.

Seu propósito é manter o PyFolds evoluindo com disciplina,
controle e rastreabilidade.
