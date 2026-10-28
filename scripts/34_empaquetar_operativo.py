"""Build the minimal operational artifact bundle.

The bundle contains M3 (hybrid ignition), M2 (fallback ignition), the
propagation model, their calibrators, thresholds and D2 prevalence metadata.
M1 is intentionally excluded: it is a diagnostic baseline kept in the
scientific archive, not an operational model.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "dist" / "operativo_v2.0.0"
REGIONS = ("chaco", "oriental")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def operational_files() -> list[Path]:
    files: list[Path] = []
    for region in REGIONS:
        files.extend(
            [
                ROOT / f"modelos/lightgbm_hibrido_{region}.booster",
                ROOT / f"modelos/lightgbm_ignicion_{region}.booster",
                ROOT / f"modelos/lightgbm_propagacion_{region}.booster",
                ROOT / f"modelos/calibradores/calibrador_hibrido_{region}.joblib",
                ROOT / f"modelos/calibradores/calibrador_ignicion_{region}.joblib",
                ROOT / f"modelos/calibradores/calibrador_propagacion_{region}.joblib",
                ROOT / f"modelos/calibradores/umbrales_hibrido_{region}.json",
                ROOT / f"modelos/calibradores/umbrales_ignicion_{region}.json",
                ROOT / f"modelos/calibradores/umbrales_propagacion_{region}.json",
            ]
        )
    files.append(ROOT / "modelos/calibradores/prevalencias_poblacionales_d2.json")
    return files


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in operational_files() if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan artefactos operativos:\n" + "\n".join(missing))

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    rows: list[tuple[str, int, str]] = []
    for src in operational_files():
        rel = src.relative_to(ROOT)
        if "m1" in rel.name.lower() or "legacy" in str(rel).lower():
            raise RuntimeError(f"Artefacto no operativo en el bundle: {rel}")
        dst = DEST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rows.append((rel.as_posix(), dst.stat().st_size, sha256(dst)))

    readme = """# Paquete operativo Incendios_PY v2.0.0

Este bundle contiene solo los artefactos necesarios para la inferencia operativa:

- M3 híbrido: `lightgbm_hibrido_*.booster`
- M2 fallback: `lightgbm_ignicion_*.booster`
- Propagación táctica: `lightgbm_propagacion_*.booster`
- Calibradores y umbrales de M3, M2 y propagación
- Prevalencias poblacionales D2

M1 **no** forma parte de este paquete. M1 es un baseline diagnóstico y permanece
en el archivo científico del candidato r4 y en el material suplementario.

Este bundle es un artefacto derivado: no reemplaza el candidato r4 ni su
manifiesto de hashes. Usar con el código del repositorio.

Verificación: `MANIFIESTO_OPERATIVO.csv` contiene tamaño y SHA-256 de cada archivo.
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")

    manifest = DEST / "MANIFIESTO_OPERATIVO.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["archivo", "bytes", "sha256"])
        writer.writerows(rows)

    total_bytes = sum(row[1] for row in rows)
    print(f"Bundle: {DEST}")
    print(f"Artefactos: {len(rows)} | bytes: {total_bytes:,}")
    print(f"Manifiesto: {manifest}")


if __name__ == "__main__":
    main()
