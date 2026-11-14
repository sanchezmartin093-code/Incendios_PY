#!/usr/bin/env bash
# run_post_render.sh — Hook de post-procesamiento editorial para Quarto
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Buscar el ejecutable de Python con dependencias (python-docx, pyyaml)
PYTHON_CMD=""

if [ -x "$PROJECT_ROOT/.pixi/envs/default/bin/python" ]; then
  PYTHON_CMD="$PROJECT_ROOT/.pixi/envs/default/bin/python"
elif [ -x "$PROJECT_ROOT/../01_manuscritos/Siniestralidad_Vial_PY/.pixi/envs/default/bin/python" ]; then
  PYTHON_CMD="$PROJECT_ROOT/../01_manuscritos/Siniestralidad_Vial_PY/.pixi/envs/default/bin/python"
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ] && "$PROJECT_ROOT/.venv/bin/python" -c "import docx, yaml" >/dev/null 2>&1; then
  PYTHON_CMD="$PROJECT_ROOT/.venv/bin/python"
elif python3 -c "import docx, yaml" >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD="uv run --with python-docx --with pyyaml python3"
fi

if [ -z "$PYTHON_CMD" ]; then
  echo "⚠️ [post-render] No se detectó un intérprete Python con python-docx/pyyaml disponible." >&2
  echo "   Instala uv (brew install uv) o ejecuta 'pixi install' para activar el formateo editorial." >&2
  exit 0
fi

# Ejecutar format_docx.py pasando cualquier argumento o en modo auto-discovery
$PYTHON_CMD "$SCRIPT_DIR/format_docx.py" "$@"
