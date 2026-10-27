"""
Script 00c: Verifies that the artifacts declared in a CSV manifest exist
and that their SHA-256 hash matches. Fails if there are missing files or altered hashes.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def verificar_manifiesto(ruta_manifiesto: Path) -> int:
    if not ruta_manifiesto.exists():
        raise FileNotFoundError(f"No existe el manifiesto: {ruta_manifiesto}")

    filas = []
    with ruta_manifiesto.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            archivo = Path(row["archivo"])
            esperado = (row.get("sha256_hash") or "").strip().lower()
            if not archivo.exists():
                estado = "AUSENTE"
                obtenido = ""
            elif not esperado or esperado == "pendiente":
                estado = "PENDIENTE"
                obtenido = ""
            else:
                obtenido = sha256(archivo)
                estado = "OK" if obtenido == esperado else "MISMATCH"
            filas.append((estado, str(archivo), esperado[:12], obtenido[:12]))

    for estado, archivo, esperado, obtenido in filas:
        print(f"{estado:<9} {archivo:<75} esperado={esperado:<12} obtenido={obtenido:<12}")

    fallos = sum(1 for estado, *_ in filas if estado in {"AUSENTE", "MISMATCH"})
    pendientes = sum(1 for estado, *_ in filas if estado == "PENDIENTE")
    print(f"\nArtefactos: {len(filas)} | fallos: {fallos} | pendientes: {pendientes}")
    return 1 if fallos else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica hashes de un manifiesto de datos.")
    parser.add_argument(
        "--manifiesto",
        type=Path,
        default=ROOT_DIR / "fuentes" / "MANIFIESTO_DATOS_v2.csv",
        help="Ruta al CSV de manifiesto.",
    )
    args = parser.parse_args()
    raise SystemExit(verificar_manifiesto(args.manifiesto))
