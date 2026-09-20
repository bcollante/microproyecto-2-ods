"""Ejecuta el notebook completo con kernel limpio y exporta los entregables.

Uso (desde la raíz del repo)::

    .venv/Scripts/python.exe scripts/run_notebook.py

Mientras corre impide que Windows suspenda el equipo (el grid search tarda
10–15 min y una suspensión hace que nbconvert agote su timeout).
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "microproyecto2_ods.ipynb"
APP = ROOT / "app" / "streamlit_app.py"
ENTREGABLES = ROOT / "entregables"
KERNEL = "microproyecto2-ods"
TIMEOUT_S = 3600


def keep_awake(enable: bool) -> None:
    if sys.platform != "win32":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def main() -> int:
    t0 = time.time()
    keep_awake(True)
    try:
        rc = subprocess.call(
            [
                sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--inplace", f"--ExecutePreprocessor.timeout={TIMEOUT_S}",
                f"--ExecutePreprocessor.kernel_name={KERNEL}", str(NB),
            ],
            cwd=ROOT,
        )
        if rc != 0:
            print(f"nbconvert terminó con código {rc}; no se exportan entregables.")
            return rc
        ENTREGABLES.mkdir(exist_ok=True)
        rc = subprocess.call(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", str(NB),
             "--output-dir", str(ENTREGABLES)],
            cwd=ROOT,
        )
        if rc != 0:
            return rc
        shutil.copy2(NB, ENTREGABLES / NB.name)
        shutil.copy2(APP, ENTREGABLES / APP.name)
    finally:
        keep_awake(False)
    print(f"Listo en {time.time() - t0:.0f} s -> {ENTREGABLES}: {NB.name}, .html y {APP.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
