# R.12: verificación en entorno limpio

- Fecha: 2026-10-28
- Python: 3.12 (venv temporal)
- Dependencias: `pip install -r requirements-lock.txt`
- Resultado: **76 passed** (2 warnings de `join_asof`)
- Hashes: `shasum -a 256 -c auditoria/HASHES.sha256` → **OK**
- Comando de pruebas: `python -m pytest -q tests`
- Nota: la instalación se hizo en un venv separado; no se reutilizó el entorno de desarrollo.
