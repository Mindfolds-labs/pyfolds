"""Android integration scaffold for Noetic + TFLite artifacts."""

from __future__ import annotations

from pathlib import Path


ANDROID_NOTES = """# Noetic Android (conceitual)

1. Converta o modelo para `.tflite` usando `pyfolds.mobile.tflite_converter`.
2. Copie o arquivo para `app/src/main/assets/model.tflite`.
3. Use TensorFlow Lite Interpreter no Android para inferência local.
4. Pré-processamento deve replicar pipeline Noetic/TF (resize + normalize).
"""


def generate_android_integration_notes(output_dir: str) -> Path:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    readme = out / "ANDROID_INTEGRATION.md"
    readme.write_text(ANDROID_NOTES, encoding="utf-8")
    return readme
