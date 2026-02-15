#!/usr/bin/env python3
"""Script para testar instalação do pyfolds em ambiente virtual limpo."""

import subprocess
import sys
import shutil
from pathlib import Path


def main() -> int:
    print("=" * 60)
    print("🧪 TESTE DE INSTALAÇÃO DO PYFOLDS")
    print("=" * 60)

    venv_path = Path("venv_test_pyfolds")
    if venv_path.exists():
        shutil.rmtree(venv_path)

    print("\n📦 1. Criando ambiente virtual...")
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

    if sys.platform == "win32":
        python_path = venv_path / "Scripts" / "python"
        pip_cmd = [str(python_path), "-m", "pip"]
    else:
        python_path = venv_path / "bin" / "python"
        pip_cmd = [str(python_path), "-m", "pip"]

    print("\n📥 2. Atualizando pip...")
    subprocess.run([*pip_cmd, "install", "--upgrade", "pip"], check=True)

    print("\n📥 3. Instalando PyTorch CPU...")
    subprocess.run(
        [
            *pip_cmd,
            "install",
            "torch",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        check=True,
    )

    print("\n📥 4. Instalando wheel do PyFolds...")
    wheels = list(Path("dist").glob("*.whl"))
    if not wheels:
        print("❌ Wheel não encontrado. Rode: python -m build")
        return 1

    wheel = wheels[0]
    subprocess.run([*pip_cmd, "install", str(wheel)], check=True)

    print("\n🔬 5. Testando importação...")
    test_code = r"""
import sys
print("🐍 Python:", sys.version.split()[0])

import torch
print("🔥 PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())

import numpy as np
print("📊 NumPy:", np.__version__)

import pyfolds
print("📦 PyFolds:", getattr(pyfolds, "__version__", "<sem __version__>"))
print("📁 Path:", pyfolds.__file__)

print("✅ IMPORTAÇÃO OK!")
"""

    result = subprocess.run(
        [str(python_path), "-c", test_code],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    if result.returncode != 0:
        print("❌ ERRO:")
        print(result.stderr)
        return 1

    print("\n🧹 6. Limpando ambiente de teste...")
    shutil.rmtree(venv_path)

    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
