#!/usr/bin/env python3
"""Build the final submission folder with individual artwork files."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENT = ROOT / "entregables"
SUB = ENT / "submission"
FIG = SUB / "figures"

ROOT_FILES = [
    (ENT / "manuscrito.docx", SUB / "manuscrito.docx"),
    (ENT / "material_suplementario.docx", SUB / "material_suplementario.docx"),
    (ENT / "highlights.docx", SUB / "highlights.docx"),
    (ENT / "cover_letter.docx", SUB / "cover_letter.docx"),
    (ENT / "figuras_maestro.pdf", SUB / "figuras_maestro.pdf"),
    (ENT / "tablas_maestro.xlsx", SUB / "tablas_maestro.xlsx"),
    (ENT / "figuras" / "graphical_abstract.pdf", SUB / "graphical_abstract.pdf"),
    (ENT / "figuras" / "graphical_abstract.png", SUB / "graphical_abstract.png"),
    (ENT / "figuras" / "graphical_abstract.tiff", SUB / "graphical_abstract.tiff"),
    (ROOT / "docs" / "CHECKLIST_ENVIO.md", SUB / "CHECKLIST_ENVIO.md"),
]

FIGURES = [
    ("Figure_1.pdf", "fig01_macro_regiones_cobertura.pdf"),
    ("Figure_2.pdf", "fig01_framework_conceptual.pdf"),
    ("Figure_3.pdf", "fig02_curvas_roc_pr_test.pdf"),
    ("Figure_4.pdf", "fig05_benchmarking_curvas_roc_pr.pdf"),
    ("Figure_5.pdf", "fig03_mapa_riesgo_operativo_nacional.pdf"),
    ("Figure_S1.png", "supp_fig01_shap.png"),
    ("Figure_S2.pdf", "supp_fig02_ablacion.pdf"),
    ("Figure_S3.pdf", "supp_fig03_cost_loss.pdf"),
    ("Figure_S4.pdf", "supp_fig04_net_benefit.pdf"),
    ("Figure_S5.pdf", "supp_fig05_rolling_origin.pdf"),
    ("Figure_S6.pdf", "supp_fig06_sensibilidad_umbral.pdf"),
    ("Figure_S7.pdf", "supp_fig07_permutation_importance.pdf"),
    ("Figure_S8.pdf", "supp_fig08_presupuesto_alertas.pdf"),
    ("Figure_S9.pdf", "supp_fig09_external_boqueron.pdf"),
    ("Figure_S10.pdf", "supp_fig10_external_formosa.pdf"),
]


def main() -> int:
    SUB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for src, dst in ROOT_FILES:
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
    for name, source in FIGURES:
        src = ENT / "figuras" / source
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, FIG / name)
    print(f"submission: {len(ROOT_FILES)} archivos + {len(FIGURES)} figuras individuales")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
