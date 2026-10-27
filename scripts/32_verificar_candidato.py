"""Script 32: read-only verification of the frozen candidate checksums."""

from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "auditoria" / "FASE5_CANDIDATO_CHECKSUMS.csv"
JSON_PATH = ROOT / "auditoria" / "FASE5_CANDIDATO_CHECKSUMS.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not CSV_PATH.exists() or not JSON_PATH.exists():
        print("Faltan los checksums del candidato.", file=sys.stderr)
        return 2

    meta = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    if not rows:
        print("El CSV de checksums está vacío.", file=sys.stderr)
        return 2

    errors = []
    total_bytes = 0
    corpus = hashlib.sha256()

    for row in rows:
        path = ROOT / row["ruta"]
        if not path.exists():
            errors.append(f"FALTA: {row['ruta']}")
            continue
        size = path.stat().st_size
        total_bytes += size
        if size != int(row["bytes"]):
            errors.append(f"TAMAÑO: {row['ruta']} ({size} != {row['bytes']})")
        digest = sha256(path)
        if digest != row["sha256"]:
            errors.append(f"HASH: {row['ruta']}")
        corpus.update(f"{row['ruta']}\t{row['sha256']}\n".encode("utf-8"))

    corpus_sha = corpus.hexdigest()
    if corpus_sha != meta["corpus_sha256"]:
        errors.append(f"CORPUS: {corpus_sha} != {meta['corpus_sha256']}")
    if len(rows) != int(meta["n_archivos"]):
        errors.append(f"CONTEO: {len(rows)} != {meta['n_archivos']}")
    if total_bytes != int(meta["bytes_totales"]):
        errors.append(f"BYTES: {total_bytes} != {meta['bytes_totales']}")

    print(f"Archivos verificados: {len(rows)}")
    print(f"Bytes verificados: {total_bytes:,}")
    print(f"Corpus SHA-256: {corpus_sha}")
    print(f"Candidato: {meta.get('tag_previsto', 'desconocido')}")
    if errors:
        print(f"ERRORES: {len(errors)}", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print("OK: checksums, tamaños y corpus coinciden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
