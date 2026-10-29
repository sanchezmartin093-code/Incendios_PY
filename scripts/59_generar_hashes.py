"""Generate auditoria/HASHES.md and HASHES.sha256."""

from pathlib import Path
import hashlib
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "auditoria"
OUT.mkdir(exist_ok=True)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        commit = commit + (" (working tree with uncommitted changes)" if dirty else "")
    except Exception:
        commit = "unknown"
    patrones = [
        "entregables/fuentes/tablas/*.csv",
        "entregables/fuentes/figuras/*.png",
        "entregables/en/*.docx",
        "entregables/es/*.docx",
        "entregables/web/*.html",
        "entregables/maestros/*.pdf",
        "entregables/maestros/*.xlsx",
        "entregables/maestros/*.png",
        "manuscrito/en/*.qmd",
        "manuscrito/en/*.md",
        "manuscrito/es/*.qmd",
        "manuscrito/assets/*.bib",
        "manuscrito/assets/*.csl",
        "scripts/*.py",
        "scripts/*.sh",
        "src/*.py",
        "README.md",
        "LICENSE",
        "entregables/README.md",
        ".vscode/settings.json",
        "_quarto.yml",
        "Makefile",
        "docs/**/*.md",
        "CITATION.cff",
        ".zenodo.json",
        "requirements-lock.txt",
        "auditoria/g0_externa/*.md",
        "auditoria/g0_externa/*.csv",
        "auditoria/g0_externa/*.json",
        "auditoria/g0_externa/*.png",
        "auditoria/g0_externa/*.parquet",
    ]
    archivos = sorted({p for patron in patrones for p in ROOT.glob(patron) if p.is_file()})
    lineas = [
        "# SHA-256 hash audit",
        "",
        f"- Generado: {datetime.now().isoformat(timespec='seconds')}",
        f"- Commit: `{commit}`",
        f"- Archivos: {len(archivos)}",
        "",
        "| Archivo | SHA-256 |",
        "|---|---|",
    ]
    sha_lines = []
    for p in archivos:
        rel = p.relative_to(ROOT)
        digest = sha(p)
        lineas.append(f"| `{rel}` | `{digest}` |")
        sha_lines.append(f"{digest}  {rel}")
    (OUT / "HASHES.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    (OUT / "HASHES.sha256").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    print(f"hashes: {len(archivos)} archivos -> auditoria/HASHES.md")


if __name__ == "__main__":
    main()
