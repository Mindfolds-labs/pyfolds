# PyFolds Icon Specification

## 1. Objetivo visual

O ícone oficial do PyFolds comunica:

- **Computação bioinspirada** (forma hexagonal modular).
- **Estrutura em camadas** (faces internas com profundidade).
- **Sinal científico/energético** (acento dourado no núcleo).

A linguagem visual prioriza leitura em tamanhos pequenos (favicon) e clareza em contextos técnicos (docs, pacote, PyPI).

## 2. Grid, proporções e área de proteção

- **Prancheta mestre:** `512 x 512`.
- **Grid base:** módulo de `32 px`.
- **Forma principal:** hexágono centrado.
- **Raio do container externo:** `96 px`.
- **Área de proteção mínima:** `48 px` ao redor do símbolo (em qualquer aplicação).
- **Escala mínima recomendada:**
  - Digital: `16 px` (favicon).
  - UI/documentação: `32 px`.
  - Material institucional: `128 px`+.

## 3. Paleta oficial (HEX / RGB)

| Token | HEX | RGB | Uso |
|---|---|---|---|
| Brand Navy 900 | `#08142D` | `8, 20, 45` | Fundo gradiente (início) |
| Brand Navy 700 | `#102B5F` | `16, 43, 95` | Fundo gradiente (fim) |
| Brand Blue 500 | `#3B7BFF` | `59, 123, 255` | Corpo principal (início) |
| Brand Blue 800 | `#1C3F8E` | `28, 63, 142` | Corpo principal (fim) |
| Brand Blue 900 | `#143574` | `20, 53, 116` | Faces internas |
| Brand Blue 600 | `#2B63D6` | `43, 99, 214` | Faces internas |
| Brand Blue 850 | `#183B86` | `24, 59, 134` | Núcleo interno |
| Brand Gold 300 | `#FFD86A` | `255, 216, 106` | Acento (início) |
| Brand Gold 600 | `#E5A800` | `229, 168, 0` | Acento (fim) |

## 4. Tipografia

Para lockups textuais da marca (quando necessário), usar:

- **Primária:** Inter / system-ui, sans-serif.
- **Peso recomendado:** 600 (títulos) e 500 (interface).
- **Tracking:** padrão (0), sem condensação.

> Nota: o arquivo de ícone oficial não embute tipografia; textos devem ser aplicados externamente conforme contexto.

## 5. Versões oficiais

Arquivos finais em `docs/_static/brand/`:

- **Light:** `pyfolds-icon-light.svg`
- **Dark:** `pyfolds-icon-dark.svg`
- **Mono:** `pyfolds-icon-mono.svg`
- **Favicon versionado:** `favicon.svg`

## 6. Pipeline de renderização (reprodutível)

Fluxo oficial:

1. **Fonte única:** `docs/brand/pyfolds-icon-master.svg`
2. **Geração de variantes SVG:** dark/light/mono
3. **Renderização raster:** PNG em múltiplos tamanhos
4. **Renderização raster local:** PNG em `_build/brand/` (não versionado)
5. **Sincronização opcional do pacote Python:** `src/pyfolds/assets/brand/` (SVG)

Comandos:

```bash
pip install cairosvg
python docs/brand/render_assets.py
```

Saídas principais:

- `docs/_static/brand/pyfolds-icon-<variant>.svg`
- `docs/_build/brand/pyfolds-icon-<variant>-<size>.png` (uso local/exportação)
- `docs/_static/brand/favicon.svg`
- `src/pyfolds/assets/brand/pyfolds-icon.svg`

## 7. Localização padronizada de assets

- **Documentação/Sphinx (canônico):** `docs/_static/brand/`
- **Especificação e pipeline:** `docs/brand/`
- **Uso no pacote Python (quando necessário):** `src/pyfolds/assets/brand/`

Evitar arquivos de marca fora desses diretórios.
