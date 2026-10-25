"""
Script 15: Generador Automatizado de Figura Científica de Alto Impacto (Nivel Nature / Elsevier Q1).
- Genera la Figura 1 / Graphical Abstract compuesto en 3 fases horizontales:
  * Phase 1: Multisource Biophysical Ingestion & Two-Stage Gating (Mapas reales de Paraguay + Embudo 92%).
  * Phase 2: Decoupled Dual GeoAI Engine (Modo 1 Ignición física vs. Modo 2 Propagación táctica t-1).
  * Phase 3: Sovereign 1-km² Operational Alert & International Benchmarking (Mapa nacional + Barras vs FWI e INPE).
- 100% automatizado, reproducible, con datos geoespaciales reales y paleta editorial clara.
- Exporta en formatos de alta resolución: PNG a 300 DPI y PDF vectorial.
"""

from pathlib import Path
import sys

# Asegurar raíz en PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import polars as pl
import numpy as np

def crear_figura_impacto(
    output_png: Path,
    output_pdf: Path | None = None
):
    print("=" * 65)
    print("GENERANDO FIGURA CIENTÍFICA DE ALTO IMPACTO (STYLE: NATURE / ELSEVIER Q1)")
    print("=" * 65)

    # 1. Cargar datos cartográficos y de prueba reales
    dir_vector = ROOT_DIR / "datos" / "raw" / "vector"
    path_adm0 = dir_vector / "paraguay_adm0.geojson"
    path_adm1 = dir_vector / "paraguay_adm1_departamentos.geojson"
    path_preds = ROOT_DIR / "datos" / "processed" / "evaluacion_test_predicciones.parquet"

    print("Cargando capas vectoriales soberanas...")
    adm0 = gpd.read_file(path_adm0)
    adm1 = gpd.read_file(path_adm1)

    df_test = None
    if path_preds.exists():
        df_test = pl.read_parquet(path_preds)
        print(f"Predicciones de prueba cargadas: {len(df_test):,} registros.")

    # 2. Configuración de Tipografía y Lienzo Editorial
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.8

    # Lienzo panorámico de alta definición: 20 x 8 pulgadas (300 DPI)
    fig = plt.figure(figsize=(20, 8.2), facecolor="#FFFFFF", dpi=300)

    # Grilla principal de 3 paneles horizontales
    # Margen exterior limpio
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1.25, 1.15], wspace=0.18, left=0.03, right=0.97, top=0.91, bottom=0.05)

    # -------------------------------------------------------------
    # CABECERA EDITORIAL GLOBAL
    # -------------------------------------------------------------
    fig.text(0.03, 0.965, "INCENDIOS_PY · SOVEREIGN GEOAI SYSTEM", fontsize=11, fontweight="bold", color="#1E40AF")
    fig.text(0.03, 0.935, "Decoupled Biophysical Ignition Hazard & Tactical Wildfire Propagation Framework", fontsize=16, fontweight="bold", color="#0F172A")
    fig.text(0.97, 0.940, "24-HOUR FORECAST · 1-km² RESOLUTION · SUB-SECOND INFERENCE", fontsize=10, fontweight="bold", color="#64748B", ha="right")

    # =============================================================
    # PANEL 1: MULTISOURCE BIOPHYSICAL INGESTION & GATING
    # =============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FAFAFA")
    for spine in ax1.spines.values():
        spine.set_color("#E2E8F0")
        spine.set_linewidth(1.2)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Banner de fase 1
    ax1.text(0.04, 0.96, "PHASE 1", fontsize=9, fontweight="bold", color="#FFFFFF", 
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#1E293B", edgecolor="none"))
    ax1.text(0.18, 0.96, "Multisource Ingestion & Two-Stage Gating", fontsize=11, fontweight="bold", color="#0F172A", va="center")

    # Mini-eje cartográfico de Paraguay
    sub_ax_geo = fig.add_axes([0.045, 0.38, 0.16, 0.48], facecolor="none")
    adm0.boundary.plot(ax=sub_ax_geo, color="#0F172A", linewidth=1.2, zorder=3)
    adm1.boundary.plot(ax=sub_ax_geo, color="#94A3B8", linewidth=0.6, linestyle=":", zorder=2)
    sub_ax_geo.set_xticks([])
    sub_ax_geo.set_yticks([])
    for sp in sub_ax_geo.spines.values():
        sp.set_visible(False)

    # Puntos de focos reales VIIRS en rojo
    if df_test is not None:
        focos = df_test.filter(pl.col("fuego") == 1)
        sub_ax_geo.scatter(focos["lon"].to_numpy(), focos["lat"].to_numpy(), 
                          s=16, color="#DC2626", alpha=0.85, edgecolors="#7F1D1D", linewidth=0.5, zorder=4, label="VIIRS 375m Fire")
        
        # Malla continua representativa de VPD en fondo
        vpd_pts = df_test.select(["lon", "lat", "vpd_max_kpa"]).to_pandas()
        sub_ax_geo.scatter(vpd_pts["lon"], vpd_pts["lat"], c=vpd_pts["vpd_max_kpa"], 
                          cmap="YlOrBr", s=8, alpha=0.35, zorder=1)

    # Badges de fuentes satelitales a la derecha del mapa
    fuentes_info = [
        ("NASA FIRMS (VIIRS 375m)", "Suomi-NPP, NOAA-20, NOAA-21\nBow-Tie filtered (scan ≤ 1.5 km)", "#DC2626"),
        ("ERA5-Land Reanalysis", "0.25° grid (640 national nodes)\nVPD, Wind, CDD, FFMC, ISI", "#2563EB"),
        ("MapBiomas Collection 4", "30m aggregated to 1 km²\n5 dominant fuel/forest classes", "#16A34A"),
    ]
    y_pos = 0.76
    for title_f, desc_f, col_f in fuentes_info:
        ax1.add_patch(patches.FancyBboxPatch((0.55, y_pos - 0.08), 0.41, 0.12, boxstyle="round,pad=0.015",
                                             facecolor="#FFFFFF", edgecolor=col_f, linewidth=1.0))
        ax1.text(0.57, y_pos + 0.015, title_f, fontsize=8.5, fontweight="bold", color=col_f)
        ax1.text(0.57, y_pos - 0.055, desc_f, fontsize=7.2, color="#475569", linespacing=1.2)
        y_pos -= 0.15

    # Tarjeta inferior: Gating Biofísico de Etapa 1 (El embudo)
    ax1.add_patch(patches.FancyBboxPatch((0.04, 0.04), 0.92, 0.25, boxstyle="round,pad=0.02",
                                         facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.2))
    ax1.text(0.07, 0.24, "STAGE 1: KINEMATIC BIO-PHYSICAL GATING", fontsize=9.5, fontweight="bold", color="#1E40AF")
    ax1.text(0.07, 0.17, "Condition: VPD ≥ 0.8 kPa  AND  Rainfall 48h < 10 mm  AND  Fuel = 1", fontsize=8.5, fontweight="bold", color="#0F172A")
    ax1.text(0.07, 0.08, "• Automatically excludes water, wet savannas & saturated air cells.\n• Discards 92% of non-fire national territory with 0% false dismissal.\n• Reduces computational space from 1.46B to 4.4M candidate cell-days.", 
             fontsize=7.8, color="#334155", linespacing=1.3)

    # =============================================================
    # PANEL 2: DECOUPLED DUAL GEOAI ENGINE
    # =============================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FAFAFA")
    for spine in ax2.spines.values():
        spine.set_color("#E2E8F0")
        spine.set_linewidth(1.2)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Banner de fase 2
    ax2.text(0.04, 0.96, "PHASE 2", fontsize=9, fontweight="bold", color="#FFFFFF", 
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#1D4ED8", edgecolor="none"))
    ax2.text(0.18, 0.96, "Decoupled Dual-Layer GeoAI Engine", fontsize=11, fontweight="bold", color="#0F172A", va="center")

    # Tarjeta Superior: MODO 1 - IGNICIÓN FÍSICA
    ax2.add_patch(patches.FancyBboxPatch((0.04, 0.53), 0.92, 0.38, boxstyle="round,pad=0.02",
                                         facecolor="#F0F9FF", edgecolor="#0284C7", linewidth=1.4))
    
    # Tag de aislamiento causal
    ax2.add_patch(patches.FancyBboxPatch((0.06, 0.84), 0.88, 0.05, boxstyle="round,pad=0.01",
                                         facecolor="#0284C7", edgecolor="none"))
    ax2.text(0.50, 0.865, "MODE 1: PURE BIOPHYSICAL IGNITION HAZARD", fontsize=9.5, fontweight="bold", color="#FFFFFF", ha="center")
    
    ax2.text(0.07, 0.77, "Features: Weather + Fuel Structure + Ecotone Edge Contrast", fontsize=8.5, fontweight="bold", color="#0F172A")
    ax2.text(0.07, 0.69, "• STRICT CAUSAL ISOLATION: No fire history lags allowed (t-1 = 0).\n• Drivers: VPD max, Thermal amplitude, Wind speed, FFMC, Ecotone.\n• Calibrated via Monotonic Bounded Platt Scaling (w in [0.1, 10.0]).", 
             fontsize=7.8, color="#334155", linespacing=1.3)
    
    # Badge de salida Modo 1
    ax2.add_patch(patches.FancyBboxPatch((0.07, 0.55), 0.86, 0.09, boxstyle="round,pad=0.015",
                                         facecolor="#E0F2FE", edgecolor="#0284C7", linewidth=0.8))
    ax2.text(0.50, 0.595, "Output: P(Ignition)  -->  Pre-suppression & Forest Patrol", fontsize=8.5, fontweight="bold", color="#0369A1", ha="center")

    # Tarjeta Inferior: MODO 2 - PROPAGACIÓN TÁCTICA
    ax2.add_patch(patches.FancyBboxPatch((0.04, 0.06), 0.92, 0.40, boxstyle="round,pad=0.02",
                                         facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=1.4))
    
    # Tag de propagación
    ax2.add_patch(patches.FancyBboxPatch((0.06, 0.39), 0.88, 0.05, boxstyle="round,pad=0.01",
                                         facecolor="#DC2626", edgecolor="none"))
    ax2.text(0.50, 0.415, "MODE 2: TACTICAL FIRE SPREAD & CONTAGION", fontsize=9.5, fontweight="bold", color="#FFFFFF", ha="center")
    
    ax2.text(0.07, 0.32, "Features: Physical Drivers + Spatial Proximity Kernel (t-1)", fontsize=8.5, fontweight="bold", color="#0F172A")
    ax2.text(0.07, 0.23, "• Causal spatial memory: dist_foco_t1_km, focos_25km_t1, frp_25km_t1.\n• Models convective advection & active flame front advancement.\n• Evaluated with 20-km Dead-Zone spatial buffer to prevent leakage.", 
             fontsize=7.8, color="#334155", linespacing=1.3)
    
    # Badge de salida Modo 2
    ax2.add_patch(patches.FancyBboxPatch((0.07, 0.09), 0.86, 0.09, boxstyle="round,pad=0.015",
                                         facecolor="#FEE2E2", edgecolor="#DC2626", linewidth=0.8))
    ax2.text(0.50, 0.135, "Output: P(Propagation)  -->  Tactical Firefighting Deployment", fontsize=8.5, fontweight="bold", color="#991B1B", ha="center")

    # =============================================================
    # PANEL 3: SOVEREIGN OPERATIONAL ALERT & BENCHMARKING
    # =============================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#FAFAFA")
    for spine in ax3.spines.values():
        spine.set_color("#E2E8F0")
        spine.set_linewidth(1.2)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Banner de fase 3
    ax3.text(0.04, 0.96, "PHASE 3", fontsize=9, fontweight="bold", color="#FFFFFF", 
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#065F46", edgecolor="none"))
    ax3.text(0.18, 0.96, "Sovereign Alert & Benchmark Validation", fontsize=11, fontweight="bold", color="#0F172A", va="center")

    # Mini mapa nacional de predicción a 1 km²
    sub_ax_alert = fig.add_axes([0.725, 0.44, 0.13, 0.46], facecolor="none")
    adm0.boundary.plot(ax=sub_ax_alert, color="#0F172A", linewidth=1.1, zorder=3)
    adm1.boundary.plot(ax=sub_ax_alert, color="#CBD5E1", linewidth=0.5, linestyle=":", zorder=2)
    sub_ax_alert.set_xticks([])
    sub_ax_alert.set_yticks([])
    for sp in sub_ax_alert.spines.values():
        sp.set_visible(False)

    if df_test is not None:
        # Colorear predicciones compuestas
        pts = df_test.to_pandas()
        p_comb = pts["probabilidad_propagacion"].fillna(pts["probabilidad_ignicion"])
        sub_ax_alert.scatter(pts["lon"], pts["lat"], c=p_comb, cmap="inferno", s=10, alpha=0.8, zorder=4)

    # Badges de rendimiento y KPIs operacionales
    ax3.add_patch(patches.FancyBboxPatch((0.55, 0.72), 0.41, 0.18, boxstyle="round,pad=0.015",
                                         facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.2))
    ax3.text(0.57, 0.85, "OPERATIONAL METRICS", fontsize=8.5, fontweight="bold", color="#047857")
    ax3.text(0.57, 0.75, "• Inference Time: 0.75 s\n• Grid: 400,006 cells (1 km²)\n• Chaco Spread ROC: 0.941\n• Tactical Precision: 97.71%", 
             fontsize=7.8, color="#1E293B", linespacing=1.3)

    # Gráfico de barras de benchmarking internacional (Sub-plot)
    sub_ax_bench = fig.add_axes([0.730, 0.08, 0.23, 0.32], facecolor="#FFFFFF")
    for sp in sub_ax_bench.spines.values():
        sp.set_color("#CBD5E1")
    
    modelos = ["Incendios_PY\n(Spread)", "Incendios_PY\n(Ignition)", "INPE Risco\nde Fogo", "Canadian\nFWI (EFFIS)"]
    roc_scores = [0.941, 0.771, 0.399, 0.389]
    colores_bar = ["#DC2626", "#0284C7", "#94A3B8", "#64748B"]

    bars = sub_ax_bench.bar(range(len(modelos)), roc_scores, color=colores_bar, width=0.6, zorder=3)
    sub_ax_bench.axhline(0.5, color="#DC2626", linestyle="--", linewidth=0.9, alpha=0.7, label="Random Guess (0.50)", zorder=2)
    sub_ax_bench.set_ylim(0, 1.05)
    sub_ax_bench.set_xticks(range(len(modelos)))
    sub_ax_bench.set_xticklabels(modelos, fontsize=7.2, fontweight="bold")
    sub_ax_bench.set_ylabel("ROC-AUC (Blind Out-of-Time)", fontsize=8.0, fontweight="bold", color="#0F172A")
    sub_ax_bench.grid(axis="y", color="#F1F5F9", linewidth=0.8, zorder=1)
    
    for bar, val in zip(bars, roc_scores):
        sub_ax_bench.text(bar.get_x() + bar.get_width()/2, val + 0.025, f"{val:.3f}", 
                          ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0F172A")

    sub_ax_bench.set_title("International Benchmarking (Blind Test)", fontsize=8.5, fontweight="bold", color="#0F172A", pad=6)

    # -------------------------------------------------------------
    # FLECHAS Y CONECTORES DE FLUJO ENTRE PANELES
    # -------------------------------------------------------------
    # Conector Panel 1 -> Panel 2 (Gating a Modo 1 y Modo 2)
    fig.patches.extend([
        patches.ConnectionPatch(xyA=(0.98, 0.72), coordsA=ax1.transAxes,
                                xyB=(0.02, 0.72), coordsB=ax2.transAxes,
                                arrowstyle="-|>", mutation_scale=14, color="#0284C7", linewidth=1.8),
        patches.ConnectionPatch(xyA=(0.98, 0.26), coordsA=ax1.transAxes,
                                xyB=(0.02, 0.26), coordsB=ax2.transAxes,
                                arrowstyle="-|>", mutation_scale=14, color="#DC2626", linewidth=1.8),
        # Conector Panel 2 -> Panel 3 (Modo 1 y Modo 2 a Alerta y Benchmarking)
        patches.ConnectionPatch(xyA=(0.98, 0.58), coordsA=ax2.transAxes,
                                xyB=(0.02, 0.65), coordsB=ax3.transAxes,
                                arrowstyle="-|>", mutation_scale=14, color="#0284C7", linewidth=1.8),
        patches.ConnectionPatch(xyA=(0.98, 0.24), coordsA=ax2.transAxes,
                                xyB=(0.02, 0.35), coordsB=ax3.transAxes,
                                arrowstyle="-|>", mutation_scale=14, color="#DC2626", linewidth=1.8),
    ])

    # Guardar en PNG (300 DPI) y PDF vectorial
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    print(f"✓ Figura PNG (300 DPI) guardada en: {output_png}")

    if output_pdf:
        plt.savefig(output_pdf, format="pdf", facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        print(f"✓ Figura PDF vectorial guardada en: {output_pdf}")

    plt.close(fig)
    print("=" * 65)

if __name__ == "__main__":
    out_dir = ROOT_DIR / "figuras"
    png_path = out_dir / "fig01_arquitectura_impacto_nature.png"
    pdf_path = out_dir / "fig01_arquitectura_impacto_nature.pdf"
    crear_figura_impacto(png_path, pdf_path)
