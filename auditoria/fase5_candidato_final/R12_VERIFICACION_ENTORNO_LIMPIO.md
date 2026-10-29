# R.12: verificación en entorno limpio

- Fecha: 2026-10-28
- Versión de metadatos: v2.1.0; el DOI de versión se asigna al depositar en Zenodo (los DOI conceptuales son 23675104/23675278).
- Python: 3.12 (venv temporal)
- Dependencias: `pip install -r requirements-lock.txt`
- Resultado: **76 passed** (2 warnings de `join_asof`)
- Hashes: `shasum -a 256 -c auditoria/HASHES.sha256` → **OK**
- Comando de pruebas: `python -m pytest -q tests`
- Nota: la instalación se hizo en un venv separado; no se reutilizó el entorno de desarrollo.
