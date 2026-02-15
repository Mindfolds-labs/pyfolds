# ARCHITECTURE_V3 — C4 (MPJRD-Wave)

## C1 — Contexto
Sistema de neurônios bio-inspirados para classificação e aprendizado online com plasticidade local e codificação temporal.

## C2 — Containers
1. **Core v2.0 (`pyfolds.core`)**
   - Sinapse, dendrito, homeostase, neuromodulação, neurônio base.
2. **Wave v3.0 (`pyfolds.wave`)**
   - Configuração e neurônio oscilatório com codificação fase/frequência.
3. **Camada/Network (`pyfolds.layers`, `pyfolds.network`)**
   - Orquestra múltiplos neurônios.
4. **Docs/Examples**
   - Fundamentação científica, matemática, e scripts de uso.

## C3 — Componentes (hierarquia funcional)
1. **Sinapse (N, I)**
   - Estado estrutural `N` (memória estável)
   - Estado volátil `I` (plasticidade curta)
2. **Dendrito (subunidade não-linear)**
   - Integração linear local `v_dend = Σ(W*x)`
   - Ativação local `a_d = sigmoid(v_dend - τ_d)`
3. **Soma (integração cooperativa)**
   - `U = Σ_d a_d` (sem WTA)
   - Spikes por limiar homeostático
4. **Axônio (oscilador de fase)**
   - Amplitude: `A = log2(1 + U)`
   - Fase: função de latência/certeza
   - Frequência: mapeada por categoria
   - Saída: componentes real/imag

## C4 — Interfaces principais
- `MPJRDWaveConfig`: parâmetros wave (`base_frequency`, `phase_decay`, etc.).
- `MPJRDWaveNeuron.forward(...)`:
  - Entradas: `x`, `reward`, `target_class`.
  - Saídas: `spikes`, `u`, `phase`, `amplitude`, `wave_real`, `wave_imag`, `phase_sync`.

## Compatibilidade
- A v3.0 é **aditiva**: módulo novo `pyfolds.wave/` sem quebrar API principal da v2.0.
