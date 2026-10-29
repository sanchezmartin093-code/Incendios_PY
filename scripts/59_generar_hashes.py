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
        "entregables/tablas/*.csv",
        "entregables/tablas/*.md",
        "entregables/figuras/*.pdf",
        "entregables/figuras/*.png",
        "entregables/*.docx",
        "entregables/*.html",
        "manuscrito/en/*.qmd",
        "manuscrito/es/*.qmd",
        "scripts/*.py",
        "src/*.py",
        "README.md",
        "CITATION.cff",
        ".zenodo.json",
        "requirements-lock.txt",
        "docs/G0_PROTOCOLO.md",
        "auditoria/G0/*.md",
        "auditoria/G0/*.csv",
        "auditoria/G0/*.json",
        "auditoria/G0/*.png",
        "auditoria/G0/*.parquet",
    ]
    archivos = sorted({p for patron in patrones for p in ROOT.glob(patron) if p.is_file()})
    lineas = [
        "# Auditoría de hashes SHA-256",
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
