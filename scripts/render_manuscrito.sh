#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

render() {
  local src="$1" outdir="$2" base="$3" to="$4"
  local nested="entregables/${src%.qmd}.${to}"
  quarto render "$src" --to "$to"
  mkdir -p "$outdir"
  mv -f "$nested" "${outdir}/${base}.${to}"
}

# English submission version (DOCX + web preview)
render manuscrito/en/manuscrito_en.qmd             entregables/en  manuscrito_en             docx
render manuscrito/en/manuscrito_en.qmd             entregables/web manuscrito_en             html
render manuscrito/en/material_suplementario_en.qmd entregables/en  material_suplementario_en docx
render manuscrito/en/material_suplementario_en.qmd entregables/web material_suplementario_en html
render manuscrito/en/cover_letter.qmd              entregables/en  cover_letter              docx
render manuscrito/en/cover_letter.qmd              entregables/web cover_letter              html

# Spanish reading version (DOCX + web preview)
render manuscrito/es/manuscrito.qmd                entregables/es  manuscrito                docx
render manuscrito/es/manuscrito.qmd                entregables/web manuscrito_es             html
render manuscrito/es/material_suplementario.qmd    entregables/es  material_suplementario    docx
render manuscrito/es/material_suplementario.qmd    entregables/web material_suplementario_es html
render manuscrito/es/title_page.qmd                entregables/es  title_page                docx
render manuscrito/es/title_page.qmd                entregables/web title_page                html

# Quarto copies external resources under output-dir; they are already embedded in DOCX/HTML.
rm -rf entregables/entregables
find entregables -mindepth 1 -type d -empty -delete
