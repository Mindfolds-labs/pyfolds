# Guide — Logging

`MPJRDNeuron` utiliza logging estruturado por nível (debug/info/warning/trace).

Boas práticas:

- `DEBUG` para inspeção de configuração e estados iniciais.
- `INFO` para mudanças de modo e eventos relevantes.
- `WARNING` para padrões anômalos (ex.: neurônio hipoativo).
- `TRACE` para métricas de passo detalhadas.
