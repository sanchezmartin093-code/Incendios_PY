#!/usr/bin/env bash
set -e

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export TZ=UTC

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "==> [Incendios_PY] Compilando entregables..."

mkdir -p entregables/en
mkdir -p entregables/es
mkdir -p entregables/fuentes/figuras

# 1. Compilación del paquete en inglés (DOCX para revista)
echo "--> Compilando documentos en inglés para envío editorial (DOCX)..."
for qmd in manuscrito/en/*.qmd; do
  if [ -f "$qmd" ]; then
    echo "    • Renderizando: $qmd..."
    quarto render "$qmd" --to docx
    base_name="$(basename "$qmd" .qmd)"
    if [ -f "entregables/manuscrito/en/${base_name}.docx" ]; then
      mv -f "entregables/manuscrito/en/${base_name}.docx" "entregables/en/${base_name}.docx"
    elif [ -f "entregables/${base_name}.docx" ]; then
      mv -f "entregables/${base_name}.docx" "entregables/en/${base_name}.docx"
    fi
  fi
done

# 2. Compilación de versión en español (HTML para lectura rápida)
echo "--> Compilando versión en español para lectura rápida (HTML)..."
for qmd in manuscrito/es/*.qmd; do
  if [ -f "$qmd" ]; then
    echo "    • Renderizando: $qmd..."
    quarto render "$qmd" --to html
    base_name="$(basename "$qmd" .qmd)"
    if [ -f "entregables/manuscrito/es/${base_name}.html" ]; then
      mv -f "entregables/manuscrito/es/${base_name}.html" "entregables/es/${base_name}.html"
    elif [ -f "entregables/${base_name}.html" ]; then
      mv -f "entregables/${base_name}.html" "entregables/es/${base_name}.html"
    fi
  fi
done

rm -rf entregables/manuscrito

echo "✓ Compilación finalizada exitosamente:"
echo "   • Paquete de envío (EN): entregables/en/"
echo "   • Versión de lectura (ES): entregables/es/"
echo "   • Figuras (EN): entregables/fuentes/figuras/"
