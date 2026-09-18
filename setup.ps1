$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    py -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name microproyecto2-ods --display-name "Python - Microproyecto 2 ODS"

Write-Host "Entorno listo. Selecciona el kernel 'Python - Microproyecto 2 ODS' en VS Code."

