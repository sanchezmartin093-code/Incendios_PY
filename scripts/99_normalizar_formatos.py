"""Enforce the repository output policy.

- Figures: only high-resolution PNG (and .gitkeep).
- Tables: only CSV.
- Audit notes: only Markdown (convert leftover .txt files).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURAS = ROOT / "entregables" / "fuentes" / "figuras"
TABLAS = ROOT / "entregables" / "fuentes" / "tablas"
AUDITORIA = ROOT / "auditoria"

PERMITIDAS_FIGURAS = {".png", ".gitkeep"}
PERMITIDAS_TABLAS = {".csv", ".gitkeep"}


def _eliminar_no_permitidos(directory: Path, permitidas: set[str]) -> int:
    eliminados = 0
    if not directory.exists():
        return 0
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() not in permitidas and path.name != ".gitkeep":
            path.unlink()
            eliminados += 1
    return eliminados


def _convertir_txt() -> int:
    convertidos = 0
    if not AUDITORIA.exists():
        return 0
    for path in AUDITORIA.rglob("*.txt"):
        destino = path.with_suffix(".md")
        if destino.exists():
            continue
        texto = path.read_text(encoding="utf-8").strip()
        if not texto.startswith("#"):
            titulo = path.stem.replace("_", " ").title()
            texto = f"# {titulo}\n\n{texto}"
        destino.write_text(texto + "\n", encoding="utf-8")
        path.unlink()
        convertidos += 1
    return convertidos


def main() -> int:
    figuras = _eliminar_no_permitidos(FIGURAS, PERMITIDAS_FIGURAS)
    tablas = _eliminar_no_permitidos(TABLAS, PERMITIDAS_TABLAS)
    txt = _convertir_txt()
    for directory in (FIGURAS, TABLAS):
        if directory.exists():
            for child in sorted(directory.rglob("*"), reverse=True):
                if child.is_dir() and not any(child.iterdir()):
                    child.rmdir()
    print(f"normalized: {figuras} figures, {tablas} tables, {txt} txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
