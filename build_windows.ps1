$ErrorActionPreference = "Stop"

python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm --clean --windowed --name GENI `
  --paths . --collect-all torch --collect-all PySide6 `
  --add-data "data\example_swedish.txt;data" run.py

Write-Host "Portable app created in dist\GENI. This script does not create an installer."
