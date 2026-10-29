#!/usr/bin/env python3
"""Build the final submission folder with individual artwork files."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENT = ROOT / "entregables"
SUB = ROOT / "dist" / "submission"
FIG = SUB / "figures"

ROOT_FILES = [
    (ENT / "en" / "manuscrito_en.docx", SUB / "manuscrito.docx"),
    (ENT / "en" / "material_suplementario_en.docx", SUB / "material_suplementario.docx"),
    (ENT / "en" / "highlights.docx", SUB / "highlights.docx"),
    (ENT / "en" / "cover_letter.docx", SUB / "cover_letter.docx"),
    (ENT / "maestros" / "figuras_maestro.pdf", SUB / "figuras_maestro.pdf"),
    (ENT / "maestros" / "tablas_maestro.xlsx", SUB / "tablas_maestro.xlsx"),
    (ENT / "maestros" / "graphical_abstract.png", SUB / "graphical_abstract.png"),
    (ROOT / "docs" / "CHECKLIST_ENVIO.md", SUB / "CHECKLIST_ENVIO.md"),
]

FIGURES = [
    ("Figure_1.png", "fig01_macro_regiones_cobertura.png"),
    ("Figure_2.png", "fig01_framework_conceptual.png"),
    ("Figure_3.png", "fig02_curvas_roc_pr_test.png"),
    ("Figure_4.png", "fig05_benchmarking_curvas_roc_pr.png"),
    ("Figure_5.png", "fig03_mapa_riesgo_operativo_nacional.png"),
    ("Figure_S1.png", "supp_fig01_shap.png"),
    ("Figure_S2.png", "supp_fig02_ablacion.png"),
    ("Figure_S3.png", "supp_fig03_cost_loss.png"),
    ("Figure_S4.png", "supp_fig04_net_benefit.png"),
    ("Figure_S5.png", "supp_fig05_rolling_origin.png"),
    ("Figure_S6.png", "supp_fig06_sensibilidad_umbral.png"),
    ("Figure_S7.png", "supp_fig07_permutation_importance.png"),
    ("Figure_S8.png", "supp_fig08_presupuesto_alertas.png"),
    ("Figure_S9.png", "supp_fig09_external_boqueron.png"),
    ("Figure_S10.png", "supp_fig10_external_formosa.png"),
]


def main() -> int:
    SUB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for src, dst in ROOT_FILES:
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
    for name, source in FIGURES:
        src = ENT / "fuentes" / "figuras" / source
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, FIG / name)
    print(f"submission: {len(ROOT_FILES)} files + {len(FIGURES)} individual figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
