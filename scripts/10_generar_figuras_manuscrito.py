"""
Script 10: High-Resolution Scientific Figure Generator for the Quarto Manuscript.
- Generates:
  * fig01_macro_regiones_cobertura: National map with Chaco/Oriental regional division and departments.
  * fig02_curvas_roc_pr_test: ROC and Precision-Recall curves with operational thresholds (0.50 vs Youden vs F0.5).
  * fig03_mapa_riesgo_operativo_nacional: National daily operational alert map at 1 km² with risk levels.
- Outputs in PNG (300 DPI) and vector PDF formats in the figuras/ folder.
"""

from pathlib import Path
import sys

# Ensure the project root is on the PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

from src.constantes import DIR_FIGURAS, DIR_PROCESSED, DIR_RAW, DIR_OUTPUTS

def generar_todas_las_figuras():
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    path_adm1 = DIR_RAW / "vector" / "paraguay_adm1_departamentos.geojson"
    if not path_adm1.exists():
        print(f"Límites vectoriales no encontrados en {path_adm1}")
        return

    gdf_deptos = gpd.read_file(path_adm1)
    gdf_deptos["departamento"] = gdf_deptos["shapeName"].str.upper().str.strip()
    gdf_deptos["region"] = gdf_deptos["departamento"].apply(
        lambda d: "Chaco" if d in {"BOQUERON", "ALTO PARAGUAY", "PRESIDENTE HAYES"} else "Oriental"
    )

    # -------------------------------------------------------------
    # FIGURE 1: Macro-Region and Territorial Coverage Map
    # -------------------------------------------------------------
    print("\n--- Generando Figura 01: Macro-Regiones y Departamentos ---")
    fig, ax = plt.subplots(figsize=(9, 10), dpi=300)

    # Colors by region
    colores_reg = {"Chaco": "#e67e22", "Oriental": "#27ae60"}
    gdf_deptos["color"] = gdf_deptos["region"].map(colores_reg)

    gdf_deptos.plot(ax=ax, color=gdf_deptos["color"], edgecolor="black", linewidth=0.8, alpha=0.6)

    # Departmental labels
    for _, row in gdf_deptos.iterrows():
        centroid = row.geometry.centroid
        ax.annotate(row["departamento"], xy=(centroid.x, centroid.y),
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    color="#2c3e50", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.65))

    patch_chaco = mpatches.Patch(facecolor="#e67e22", edgecolor="black", alpha=0.6, label="Chaco (western region, 240,880 km²)")
    patch_oriental = mpatches.Patch(facecolor="#27ae60", edgecolor="black", alpha=0.6, label="Oriental Region (159,126 km²)")
    ax.legend(handles=[patch_chaco, patch_oriental], loc="lower left", fontsize=9, frameon=True)

    ax.set_title("Study domain and macro-regional stratification of Paraguay\n1-km² regular grid (EPSG:32721)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude (°W)", fontsize=9)
    ax.set_ylabel("Latitude (°S)", fontsize=9)
    plt.tight_layout()

    fig.savefig(DIR_FIGURAS / "fig01_macro_regiones_cobertura.png", dpi=300)
    fig.savefig(DIR_FIGURAS / "fig01_macro_regiones_cobertura.pdf")
    plt.close(fig)
    print(" ✓ Fig 01 guardada en PNG y PDF.")

    # -------------------------------------------------------------
    # FIGURE 2: ROC and Precision-Recall Curves (Blind Test 2024–2025)
    # -------------------------------------------------------------
    print("\n--- Generando Figura 02: Curvas ROC y PR en Test 2024–2025 (Dual) ---")
    p_ign = DIR_PROCESSED / "evaluacion_test_ignicion.parquet"
    p_prop = DIR_PROCESSED / "evaluacion_test_propagacion.parquet"
    if p_ign.exists() and p_prop.exists():
        df_ign = pl.read_parquet(p_ign)
        df_prop = pl.read_parquet(p_prop)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

        modelos_curvas = [
            ("Chaco", "Hybrid ignition M3", df_ign, "target_ignicion", "probabilidad_hibrido", "#d35400", "-"),
            ("Chaco", "Tactical propagation", df_prop, "target_propagacion", "probabilidad_propagacion", "#c0392b", "--"),
            ("Oriental", "Hybrid ignition M3", df_ign, "target_ignicion", "probabilidad_hibrido", "#16a085", "-"),
            ("Oriental", "Tactical propagation", df_prop, "target_propagacion", "probabilidad_propagacion", "#27ae60", "--"),
        ]

        for reg, capa, df_fuente, col_y, col_prob, col, ls in modelos_curvas:
            sub = df_fuente.filter(pl.col("region") == reg)
            y_true = sub[col_y].to_numpy()
            y_prob = sub[col_prob].to_numpy()

            if len(np.unique(y_true)) > 1:
                # ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_score = auc(fpr, tpr)
                ax1.plot(fpr, tpr, color=col, linestyle=ls, linewidth=2.0, label=f"{reg} - {capa} (AUC = {roc_score:.3f})")

                # Precision-Recall curve
                prec, rec, _ = precision_recall_curve(y_true, y_prob)
                pr_score = average_precision_score(y_true, y_prob)
                ax2.plot(rec, prec, color=col, linestyle=ls, linewidth=2.0, label=f"{reg} - {capa} (PR-AUC = {pr_score:.3f})")

        ax1.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Chance (AUC = 0.500)")
        ax1.set_title("ROC curves (receiver operating characteristic)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("False positive rate (1 − specificity)", fontsize=10)
        ax1.set_ylabel("True positive rate (recall)", fontsize=10)
        ax1.legend(loc="lower right", fontsize=8.5)

        ax2.set_title("Precision–recall curves (rare-event detection)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Recall", fontsize=10)
        ax2.set_ylabel("Precision (PPV)", fontsize=10)
        ax2.legend(loc="upper right", fontsize=8.5)

        plt.suptitle("Retrospective out-of-time performance (2024–2025)",
                     fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()

        fig.savefig(DIR_FIGURAS / "fig02_curvas_roc_pr_test.png", dpi=300)
        fig.savefig(DIR_FIGURAS / "fig02_curvas_roc_pr_test.pdf")
        plt.close(fig)
        print(" ✓ Fig 02 guardada en PNG y PDF con 4 curvas.")

    # -------------------------------------------------------------
    # FIGURE 3: National Operational Fire Risk Map
    # -------------------------------------------------------------
    print("\n--- Generando Figura 03: Mapa Operativo Nacional de Riesgo ---")
    dir_raster = DIR_OUTPUTS / "raster_riesgo_diario"
    # Fixed date for a reproducible, non-cherry-picked alert map. The date is
    # regenerated with the frozen model before this script is run.
    fecha_alerta = "2024-09-09"
    p_alerta = dir_raster / f"alerta_nacional_{fecha_alerta}.parquet"
    if not p_alerta.exists():
        archivos_alerta = sorted(list(dir_raster.glob("alerta_nacional_*.parquet")))
        if not archivos_alerta:
            print(f"No alert layers found in {dir_raster}")
            return
        p_alerta = archivos_alerta[-1]
        fecha_alerta = p_alerta.stem.replace("alerta_nacional_", "")
    if p_alerta.exists():
        df_alerta = pl.read_parquet(p_alerta)

        # Representative sampling of cells for fast cartographic rendering (1 out of every 3 cells)
        df_plot = df_alerta.gather_every(3)

        fig, ax = plt.subplots(figsize=(9, 10), dpi=300)
        gdf_deptos.boundary.plot(ax=ax, color="#7f8c8d", linewidth=0.7, zorder=3)

        # Color points according to risk level
        colores_nivel = {
            "EXTREMO": "#962d2d",
            "ALTO": "#e74c3c",
            "MODERADO": "#f39c12",
            "BAJO": "#bdc3c7",
            "NULO": "#ecf0f1"
        }

        # Plot by layers so that higher risk levels appear on top
        for nivel in ["BAJO", "MODERADO", "ALTO", "EXTREMO"]:
            sub_n = df_plot.filter(pl.col("nivel_alerta") == nivel)
            if sub_n.height > 0:
                ax.scatter(
                    sub_n["lon"].to_numpy(),
                    sub_n["lat"].to_numpy(),
                    c=colores_nivel[nivel],
                    s=1.2,
                    alpha=0.7 if nivel in ["ALTO", "EXTREMO"] else 0.35,
                    label=f"{ {'EXTREMO':'Extreme','ALTO':'High','MODERADO':'Moderate','BAJO':'Low','NULO':'None'}[nivel] } ({df_alerta.filter(pl.col('nivel_alerta') == nivel).height:,} km²)",
                    zorder=2 if nivel in ["ALTO", "EXTREMO"] else 1
                )

        ax.set_title(f"National operational fire alert map\nDate: {fecha_alerta} | Resolution: 1 km² (Incendios_PY)",
                     fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Longitude (°W)", fontsize=9)
        ax.set_ylabel("Latitude (°S)", fontsize=9)
        ax.legend(loc="lower left", fontsize=8.5, frameon=True, title="Alert level")
        plt.tight_layout()

        fig.savefig(DIR_FIGURAS / "fig03_mapa_riesgo_operativo_nacional.png", dpi=300)
        fig.savefig(DIR_FIGURAS / "fig03_mapa_riesgo_operativo_nacional.pdf")
        plt.close(fig)
        print(" ✓ Fig 03 guardada en PNG y PDF.")

    print("\n✓ Proceso de generación de figuras del manuscrito completado con éxito.")

if __name__ == "__main__":
    generar_todas_las_figuras()
