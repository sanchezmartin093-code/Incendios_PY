"""
Script 31: Freeze the scientific submission candidate.

Computes SHA-256 checksums for the frozen scientific assets:
datasets, targets, models, calibrators, thresholds, prevalence metadata,
selection tables, and the confirmation holdout.

The manuscript, release metadata and submission files are intentionally not
part of this checksum corpus: they will change after Phase 5 confirmation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import subprocess

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "auditoria" / "FASE5_CANDIDATO_CHECKSUMS.csv"
OUT_JSON = ROOT / "auditoria" / "FASE5_CANDIDATO_CHECKSUMS.json"
OUT_ARCHIVE = ROOT / "auditoria" / "FASE5_CANDIDATO_GIT_ARCHIVE.sha256"

REGIONES_MODELO = ("chaco", "oriental")
MODOS_MODELO = ("hibrido", "ignicion", "propagacion", "m1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def collect_paths() -> list[tuple[str, Path, str]]:
    entries: list[tuple[str, Path, str]] = []

    def add(category: str, pattern: str, note: str = "") -> None:
        for p in sorted(ROOT.glob(pattern)):
            if p.is_file():
                entries.append((category, p, note))

    # Core processed data
    for name in [
        "grilla_nacional_1km.parquet",
        "clima_era5_diario.parquet",
        "clima_era5_diario.meta.json",
        "combustible_mapbiomas_1km_anual.parquet",
        "combustible_mapbiomas_1km_anual.meta.json",
        "desmonte_hansen_1km.parquet",
        "focos_viirs_diarios.parquet",
        "fwi_completo_diario.parquet",
        "inpe_diario.parquet",
        "conteos_poblacionales_d2.parquet",
        "plan_muestreo_d2.parquet",
    ]:
        add("processed", f"datos/processed/{name}")

    # Deterministic annual dataset and manifest
    add("dataset", "datos/processed/dataset_unificado_candidatos/dataset_*.parquet")
    add("dataset", "datos/processed/dataset_unificado_candidatos/manifest.json")

    # Canonical modelling targets
    add("targets", "datos/processed/targets_poblacionales/*_ignicion.parquet")
    add("targets", "datos/processed/targets_poblacionales/*_propagacion.parquet")

    # Confirmation holdout: must remain untouched until Phase 5
    add("holdout_confirmacion", "datos/processed/holdout_confirmacion/*", "No usar hasta Fase 5")

    # Models, calibrators, thresholds and prevalence metadata
    add("models", "modelos/*.booster")
    add("models", "modelos/calibradores/*.joblib")
    add("models", "modelos/calibradores/umbrales_*.json")
    add("models", "modelos/calibradores/prevalencias_poblacionales_d2.json")

    # Selection and decision evidence
    add("selection", "tablas/tabla10_validacion_2023*.csv")
    add("selection", "tablas/tabla10_comparaciones_*.csv")
    add("selection", "tablas/tabla08b_*.csv")
    add("selection", "tablas/tabla02_*.csv")
    add("selection", "tablas/tabla04_*.csv")
    add("selection", "tablas/tabla05_*.csv")
    add("selection", "tablas/tabla07_*.csv")
    add("selection", "tablas/tabla09_*.csv")
    add("decision", "auditoria/FASE2_DECISION_MODELO.md")
    add("decision", "auditoria/FASE2_DECISION_MODELO.json")
    add("decision", "auditoria/FASE2_PROTOCOLO_SELECCION.md")
    add("decision", "auditoria/FASE2_AUDITORIA_MODELOS.md")
    add("decision", "auditoria/FASE2_DATASET_MANIFEST.json")
    add("decision", "auditoria/FASE2_HOLDOUT_META.json")
    add("decision", "auditoria/FASE2_CLIMA_PROVENANCE.csv")
    add("decision", "auditoria/FASE2_REPRO_BUILD_CHECK.txt")

    # Code contract and environment
    add("code", "scripts/31_congelar_candidato.py")
    add("code", "scripts/32_verificar_candidato.py")
    add("code", "scripts/33_verificar_trazabilidad_git.py")
    add("code", "src/modelo.py")
    add("code", "src/alerta.py")
    add("code", "src/holdout.py")
    add("code", "scripts/26_validacion_2023.py")
    add("code", "scripts/27_validacion_espacial_2023.py")
    add("code", "scripts/29_lead_time_2023.py")
    add("code", "scripts/30_validacion_propagacion_2023.py")
    add("code", "tests/test_alerta.py")
    add("code", "tests/test_holdout_confirmacion.py")
    add("environment", "requirements-lock.txt")
    add("environment", "pyproject.toml")
    add("environment", ".github/workflows/ci.yml")

    # Deduplicate by resolved path, preserve first category
    seen = set()
    out = []
    for category, path, note in entries:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append((category, path, note))
    return out


def required_artifacts() -> list[Path]:
    paths = [
        ROOT / "datos/processed/holdout_confirmacion/celdas_holdout.parquet",
        ROOT / "datos/processed/holdout_confirmacion/celdas_buffer.parquet",
        ROOT / "datos/processed/targets_poblacionales/train_ignicion.parquet",
        ROOT / "datos/processed/targets_poblacionales/val_ignicion.parquet",
        ROOT / "datos/processed/targets_poblacionales/test_ignicion.parquet",
        ROOT / "modelos/calibradores/prevalencias_poblacionales_d2.json",
        ROOT / "scripts/33_verificar_trazabilidad_git.py",
        ROOT / "requirements-lock.txt",
    ]
    for region in REGIONES_MODELO:
        for modo in MODOS_MODELO:
            paths.extend([
                ROOT / f"modelos/lightgbm_{modo}_{region}.booster",
                ROOT / f"modelos/calibradores/calibrador_{modo}_{region}.joblib",
                ROOT / f"modelos/calibradores/umbrales_{modo}_{region}.json",
            ])
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        required=True,
        help="Tag previsto para el candidato, por ejemplo submission-candidate-2026-10-27-r3",
    )
    return parser.parse_args()


def main() -> None:
    tag_previsto = parse_args().tag
    if git("status", "--porcelain"):
        raise RuntimeError(
            "El repositorio no está limpio. Congelá el candidato desde un commit limpio."
        )

    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", "HEAD"], cwd=ROOT
    )
    archive_sha = hashlib.sha256(archive).hexdigest()
    OUT_ARCHIVE.write_text(
        f"{archive_sha}  git-archive-source-{commit[:7]}.tar\n",
        encoding="utf-8",
    )

    rows = []
    for category, path, note in collect_paths():
        rel = path.relative_to(ROOT).as_posix()
        rows.append({
            "categoria": category,
            "ruta": rel,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "nota": note,
        })

    missing = [
        p.relative_to(ROOT).as_posix() for p in required_artifacts() if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Faltan artefactos críticos: {missing}")

    corpus = hashlib.sha256()
    for row in rows:
        corpus.update(f"{row['ruta']}\t{row['sha256']}\n".encode("utf-8"))
    corpus_sha = corpus.hexdigest()

    metadata = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "source_commit": commit,
        "source_tree": tree,
        "source_archive_sha256": archive_sha,
        "source_status": "clean",
        "n_archivos": len(rows),
        "bytes_totales": sum(r["bytes"] for r in rows),
        "corpus_sha256": corpus_sha,
        "tag_previsto": tag_previsto,
        "alcance": (
            "Candidato científico: datos procesados, targets, modelos, "
            "calibradores, umbrales, prevalencias, tablas de selección, "
            "holdout de confirmación, código y lock. El manuscrito y los "
            "metadatos de release no están congelados."
        ),
        "regla": "No reentrenar, reajustar umbrales ni tocar el holdout después de este tag.",
    }

    OUT_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["categoria", "ruta", "bytes", "sha256", "nota"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Commit: {commit}")
    print(f"Tree: {tree}")
    print(f"Git archive SHA-256: {archive_sha}")
    print(f"Archivos: {len(rows)}")
    print(f"Bytes: {sum(r['bytes'] for r in rows):,}")
    print(f"Corpus SHA-256: {corpus_sha}")
    print(f"CSV: {OUT_CSV}")
    print(f"JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
