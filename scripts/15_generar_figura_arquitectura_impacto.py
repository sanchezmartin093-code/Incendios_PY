"""
Script 15: Automated Generator of High-Impact Scientific Figure (Nature / Elsevier Q1 Level).
- Layout issue fixes:
  1. MapBiomas Box: dimension computed with bottom padding (y >= 0.03) to avoid edge clipping.
  2. Operational Metrics Box: redesigned with balanced proportions and structured KPI cards.
  3. Flow arrows: 100% HORIZONTAL (slope 0) connected in the inter-panel gutters without crossing axes or text.
- Exports in high-resolution formats: PNG at 300 DPI and vector PDF.
"""

from pathlib import Path
import sys

# Ensure root in PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import geopandas as gpd
import polars as pl
import pandas as pd
import numpy as np

def crear_figura_impacto(
    output_png: Path,
    output_pdf: Path | None = None
):
    print("=" * 65)
    print("GENERANDO FIGURA CIENTÍFICA (LAYOUT CORREGIDO: FLECHAS HORIZONTALES Y SIN CORTES)")
    print("=" * 65)

    # 1. Load real cartographic and test data
    dir_vector = ROOT_DIR / "datos" / "raw" / "vector"
    path_adm0 = dir_vector / "paraguay_adm0.geojson"
    path_adm1 = dir_vector / "paraguay_adm1_departamentos.geojson"
    path_preds = ROOT_DIR / "datos" / "processed" / "evaluacion_test_ignicion.parquet"

    print("Cargando capas vectoriales soberanas...")
    adm0 = gpd.read_file(path_adm0)
    adm1 = gpd.read_file(path_adm1)

    df_test = None
    if path_preds.exists():
        df_test = pl.read_parquet(path_preds)
        print(f"Predicciones de prueba cargadas: {len(df_test):,} registros.")

    # Dynamic metrics from the generated tables (avoids hardcoded KPIs).
    def _tabla(nombre: str) -> pd.DataFrame | None:
        ruta = ROOT_DIR / "entregables" / "tablas" / nombre
        return pd.read_csv(ruta) if ruta.exists() else None

    t04 = _tabla("tabla04_benchmarking_internacional.csv")
    t07 = _tabla("tabla07_evaluacion_poblacional.csv")
    tS4 = _tabla("tablaS4_perfil_computacional.csv")

    def _roc(tabla, region: str, patron: str, defecto: float) -> float:
        if tabla is None or "modelo" not in tabla.columns:
            return defecto
        sub = tabla[(tabla["region"] == region) & tabla["modelo"].str.contains(patron, regex=False)]
        return float(sub["roc_auc"].iloc[0]) if len(sub) else defecto

    roc_m3_chaco = _roc(t04, "Chaco", "Híbrida M3", 0.786)
    roc_m3_oriental = _roc(t04, "Oriental", "Híbrida M3", 0.828)
    roc_prop_chaco = _roc(t04, "Chaco", "Propagación Táctica", 0.912)
    roc_fwi_chaco = _roc(t04, "Chaco", "Canadian FWI", 0.518)
    roc_inpe_chaco = _roc(t04, "Chaco", "Proxy INPE", 0.504)

    precision_prop_pob = 11.87
    if t07 is not None and "modelo" in t07.columns:
        sub = t07[(t07["region"] == "Chaco") & (t07["modelo"] == "M2_propagacion_tactica")]
        if len(sub):
            precision_prop_pob = float(sub["precision_poblacional_pct"].iloc[0])

    latency_s = 1.77
    if tS4 is not None and "etapa" in tS4.columns:
        sub = tS4[tS4["etapa"].str.contains("TOTAL", na=False)]
        if len(sub):
            latency_s = float(sub["latencia_arm64_s"].iloc[0])

    # 2. Editorial Typography and Style Configuration
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.8

    # High-definition panoramic canvas: 20 x 8.5 inches (300 DPI)
    fig = plt.figure(figsize=(20, 8.5), facecolor="#FFFFFF", dpi=300)

    # -------------------------------------------------------------
    # GLOBAL EDITORIAL HEADER
    # -------------------------------------------------------------
    fig.text(0.03, 0.965, "INCENDIOS_PY · SOVEREIGN GEOAI SYSTEM", fontsize=11, fontweight="bold", color="#1E40AF")
    fig.text(0.03, 0.932, "Decoupled Biophysical Ignition Hazard & Tactical Wildfire Propagation Framework", fontsize=16, fontweight="bold", color="#0F172A")
    fig.text(0.97, 0.938, "DAILY HAZARD NOWCAST · 1-km² RESOLUTION", fontsize=10, fontweight="bold", color="#64748B", ha="right")

    # Main grid of 3 horizontal panels
    # Wide gutter between panels to give space to arrows and clean horizontal labels
    gs_main = gridspec.GridSpec(1, 3, width_ratios=[1.18, 1.25, 1.18], wspace=0.18, 
                                left=0.03, right=0.97, top=0.90, bottom=0.05)

    # =============================================================
    # PANEL 1: MULTISOURCE BIOPHYSICAL INGESTION & GATING
    # =============================================================
    ax1_bg = fig.add_subplot(gs_main[0, 0])
    ax1_bg.set_facecolor("#FAFAFA")
    for spine in ax1_bg.spines.values():
        spine.set_color("#E2E8F0")
        spine.set_linewidth(1.2)
    ax1_bg.set_xticks([])
    ax1_bg.set_yticks([])

    # Top banner of phase 1
    ax1_bg.text(0.04, 0.96, "PHASE 1", fontsize=8.5, fontweight="bold", color="#FFFFFF", 
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1E293B", edgecolor="none"))
    ax1_bg.text(0.19, 0.96, "Multisource Ingestion & Two-Stage Gating", fontsize=9.8, fontweight="bold", color="#0F172A", va="center")

    # Panel 1 internal sub-grid:
    # Row 0: Map on the left (43%) and 3 cards on the right (57%)
    # Row 1: Horizontal biophysical gating at the bottom
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[0, 0], 
                                          height_ratios=[0.63, 0.37], width_ratios=[0.43, 0.57], 
                                          wspace=0.10, hspace=0.12)

    # Sub-ax 1A: Map of Paraguay
    ax1_map = fig.add_subplot(gs1[0, 0])
    ax1_map.set_facecolor("none")
    adm0.boundary.plot(ax=ax1_map, color="#0F172A", linewidth=1.2, zorder=3)
    adm1.boundary.plot(ax=ax1_map, color="#94A3B8", linewidth=0.5, linestyle=":", zorder=2)
    
    # Strict limits to confine the map to its column
    ax1_map.set_xlim(-63.0, -53.8)
    ax1_map.set_ylim(-28.0, -18.8)
    ax1_map.set_xticks([])
    ax1_map.set_yticks([])
    for sp in ax1_map.spines.values():
        sp.set_visible(False)

    if df_test is not None:
        # Continuous representative VPD grid in the background
        vpd_pts = df_test.select(["lon", "lat", "vpd_max_kpa"]).to_pandas()
        ax1_map.scatter(vpd_pts["lon"], vpd_pts["lat"], c=vpd_pts["vpd_max_kpa"], 
                        cmap="YlOrBr", s=7, alpha=0.35, zorder=1)
        
        # Real VIIRS fire hotspot points in red
        focos = df_test.filter(pl.col("fuego") == 1)
        ax1_map.scatter(focos["lon"].to_numpy(), focos["lat"].to_numpy(), 
                        s=16, color="#DC2626", alpha=0.90, edgecolors="#7F1D1D", linewidth=0.5, zorder=4)

    ax1_map.set_title("Paraguay (1 km² Grid)\nVIIRS Fire Hotspots", fontsize=8.0, fontweight="bold", color="#0F172A", pad=2)

    # Sub-ax 1B: Three Satellite Source Cards (Evenly distributed without clipping)
    ax1_cards = fig.add_subplot(gs1[0, 1])
    ax1_cards.set_facecolor("none")
    ax1_cards.set_xlim(0, 1)
    ax1_cards.set_ylim(0, 1)
    ax1_cards.set_xticks([])
    ax1_cards.set_yticks([])
    for sp in ax1_cards.spines.values():
        sp.set_visible(False)

    fuentes_info = [
        ("NASA FIRMS (VIIRS 375m)", "Suomi-NPP, NOAA-20, NOAA-21\nBow-Tie filtered (scan ≤ 1.5 km)", "#DC2626", "#FEF2F2"),
        ("ERA5-Land Reanalysis", "0.25° grid (640 national nodes)\nVPD, Wind, CDD, FFMC, ISI", "#2563EB", "#EFF6FF"),
        ("Fuel Cover (MapBiomas)", "Annual C2, prior year\n400.006 cells", "#16A34A", "#F0FDF4"),
    ]
    
    # 3 cards with height 0.27 and spacing of 0.05 (from y=0.04 to y=0.95 -> 0% bottom clipping)
    card_height = 0.26
    card_bottoms = [0.68, 0.36, 0.04]
    for (title_f, desc_f, col_f, bg_f), y_b in zip(fuentes_info, card_bottoms):
        ax1_cards.add_patch(patches.FancyBboxPatch((0.02, y_b), 0.96, card_height, boxstyle="round,pad=0.02",
                                                   facecolor=bg_f, edgecolor=col_f, linewidth=1.1))
        ax1_cards.text(0.06, y_b + card_height - 0.07, title_f, fontsize=8.2, fontweight="bold", color=col_f)
        ax1_cards.text(0.06, y_b + card_height - 0.20, desc_f, fontsize=7.2, color="#334155", linespacing=1.25)

    # Sub-ax 1C: Stage 1 Biophysical Gating Funnel
    ax1_gate = fig.add_subplot(gs1[1, :])
    ax1_gate.set_facecolor("none")
    ax1_gate.set_xlim(0, 1)
    ax1_gate.set_ylim(0, 1)
    ax1_gate.set_xticks([])
    ax1_gate.set_yticks([])
    for sp in ax1_gate.spines.values():
        sp.set_visible(False)

    ax1_gate.add_patch(patches.FancyBboxPatch((0.02, 0.05), 0.96, 0.90, boxstyle="round,pad=0.02",
                                              facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.2))
    ax1_gate.text(0.05, 0.77, "STAGE 1: KINEMATIC BIO-PHYSICAL GATING", fontsize=9.2, fontweight="bold", color="#1E40AF")
    ax1_gate.text(0.05, 0.53, "Condition: VPD ≥ 0.8 kPa  AND  Rainfall 48h < 10 mm  AND  Fuel = 1", fontsize=8.2, fontweight="bold", color="#0F172A")
    ax1_gate.text(0.05, 0.18, "• Automatically excludes water, wet savannas & saturated air cells.\n• Discards 39.2% of non-fire national cell-days (89.94% fire retention).\n• Focuses compute from 1,168.8M cell-days to high-risk candidates.", 
                  fontsize=7.4, color="#334155", linespacing=1.3)

    # =============================================================
    # PANEL 2: DECOUPLED DUAL GEOAI ENGINE
    # =============================================================
    ax2_bg = fig.add_subplot(gs_main[0, 1])
    ax2_bg.set_facecolor("#FAFAFA")
    for spine in ax2_bg.spines.values():
        spine.set_color("#E2E8F0")
        spine.set_linewidth(1.2)
    ax2_bg.set_xticks([])
    ax2_bg.set_yticks([])

    # Top banner of phase 2
    ax2_bg.text(0.04, 0.96, "PHASE 2", fontsize=8.5, fontweight="bold", color="#FFFFFF", 
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1D4ED8", edgecolor="none"))
    ax2_bg.text(0.19, 0.96, "Decoupled Dual-Layer GeoAI Engine", fontsize=9.8, fontweight="bold", color="#0F172A", va="center")

    # Top Card: MODE 1 - MULTISCALE HYBRID IGNITION M3 (Centered at y = 0.72)
    ax2_bg.add_patch(patches.FancyBboxPatch((0.04, 0.53), 0.92, 0.38, boxstyle="round,pad=0.02",
                                            facecolor="#F0F9FF", edgecolor="#0284C7", linewidth=1.4))
    
    # Causal isolation tag
    ax2_bg.add_patch(patches.FancyBboxPatch((0.06, 0.84), 0.88, 0.05, boxstyle="round,pad=0.01",
                                            facecolor="#0284C7", edgecolor="none"))
    ax2_bg.text(0.50, 0.865, "MODE 1: MULTISCALE SOCIO-ECOLOGICAL HYBRID (M3)", fontsize=8.8, fontweight="bold", color="#FFFFFF", ha="center")
    
    ax2_bg.text(0.07, 0.77, "Features: Weather + Fuel Structure + Anthropogenic Priors (10/25km)", fontsize=8.2, fontweight="bold", color="#0F172A")
    ax2_bg.text(0.07, 0.69, "• STRICT CAUSAL ISOLATION: No fire history lags allowed (t-1 = 0).\n• Drivers: VPD max, Thermal amplitude, Wind speed, Pasture fire pressure.\n• Calibrated via Monotonic Platt Scaling & Split Conformal Prediction.", 
                fontsize=7.6, color="#334155", linespacing=1.3)
    
    # Mode 1 output badge (Aligned at y = 0.595)
    ax2_bg.add_patch(patches.FancyBboxPatch((0.07, 0.55), 0.86, 0.09, boxstyle="round,pad=0.015",
                                            facecolor="#E0F2FE", edgecolor="#0284C7", linewidth=0.8))
    ax2_bg.text(0.50, 0.595, "Output: P(Ignition M3)  -->  Pre-suppression & Forest Patrol", fontsize=8.4, fontweight="bold", color="#0369A1", ha="center")

    # Bottom Card: MODE 2 - TACTICAL PROPAGATION (Centered at y = 0.26)
    ax2_bg.add_patch(patches.FancyBboxPatch((0.04, 0.06), 0.92, 0.40, boxstyle="round,pad=0.02",
                                            facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=1.4))
    
    # Propagation tag
    ax2_bg.add_patch(patches.FancyBboxPatch((0.06, 0.39), 0.88, 0.05, boxstyle="round,pad=0.01",
                                            facecolor="#DC2626", edgecolor="none"))
    ax2_bg.text(0.50, 0.415, "MODE 2: TACTICAL FIRE SPREAD & CONTAGION", fontsize=9.2, fontweight="bold", color="#FFFFFF", ha="center")
    
    ax2_bg.text(0.07, 0.32, "Features: Physical Drivers + Spatial Proximity Kernel (t-1)", fontsize=8.5, fontweight="bold", color="#0F172A")
    ax2_bg.text(0.07, 0.23, "• Causal spatial memory: dist_foco_t1_km, focos_25km_t1, frp_25km_t1.\n• Models convective advection & active flame front advancement.\n• Evaluated with 20-km Dead-Zone spatial buffer to prevent leakage.", 
                fontsize=7.6, color="#334155", linespacing=1.3)
    
    # Mode 2 output badge (Aligned at y = 0.135)
    ax2_bg.add_patch(patches.FancyBboxPatch((0.07, 0.09), 0.86, 0.09, boxstyle="round,pad=0.015",
                                            facecolor="#FEE2E2", edgecolor="#DC2626", linewidth=0.8))
    ax2_bg.text(0.50, 0.135, "Output: P(Propagation)  -->  Tactical Firefighting Deployment", fontsize=8.4, fontweight="bold", color="#991B1B", ha="center")

    # =============================================================
    # PANEL 3: SOVEREIGN OPERATIONAL ALERT & BENCHMARKING
    # =============================================================
    ax3_bg = fig.add_subplot(gs_main[0, 2])
    ax3_bg.set_facecolor("#FAFAFA")
    for spine in ax3_bg.spines.values():
        spine.set_color("#E2E8F0")
        spine.set_linewidth(1.2)
    ax3_bg.set_xticks([])
    ax3_bg.set_yticks([])

    # Top banner of phase 3
    ax3_bg.text(0.04, 0.96, "PHASE 3", fontsize=8.5, fontweight="bold", color="#FFFFFF", 
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#065F46", edgecolor="none"))
    ax3_bg.text(0.19, 0.96, "Sovereign Alert & Benchmark Validation", fontsize=9.8, fontweight="bold", color="#0F172A", va="center")

    # Panel 3 internal sub-grid:
    # Row 0: Map on the left (45%) and structured KPIs on the right (55%)
    # Row 1: International benchmarking at the bottom (full width)
    gs3 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[0, 2], 
                                          height_ratios=[0.56, 0.44], width_ratios=[0.45, 0.55], 
                                          wspace=0.10, hspace=0.14)

    # Sub-ax 3A: National Prediction Map
    ax3_map = fig.add_subplot(gs3[0, 0])
    ax3_map.set_facecolor("none")
    adm0.boundary.plot(ax=ax3_map, color="#0F172A", linewidth=1.1, zorder=3)
    adm1.boundary.plot(ax=ax3_map, color="#CBD5E1", linewidth=0.5, linestyle=":", zorder=2)
    ax3_map.set_xlim(-63.0, -53.8)
    ax3_map.set_ylim(-28.0, -18.8)
    ax3_map.set_xticks([])
    ax3_map.set_yticks([])
    for sp in ax3_map.spines.values():
        sp.set_visible(False)

    if df_test is not None:
        pts = df_test.to_pandas()
        prob_ign_col = "probabilidad_hibrido" if "probabilidad_hibrido" in pts.columns else "probabilidad_calibrada"
        p_comb = pts[prob_ign_col]
        ax3_map.scatter(pts["lon"], pts["lat"], c=p_comb, cmap="inferno", s=8, alpha=0.85, zorder=4)

    ax3_map.set_title("Operational Alert Map\n(1 km² Resolution)", fontsize=8.0, fontweight="bold", color="#0F172A", pad=2)

    # Sub-ax 3B: Operational KPIs (Redesigned proportionally without oversizing)
    ax3_kpis = fig.add_subplot(gs3[0, 1])
    ax3_kpis.set_facecolor("none")
    ax3_kpis.set_xlim(0, 1)
    ax3_kpis.set_ylim(0, 1)
    ax3_kpis.set_xticks([])
    ax3_kpis.set_yticks([])
    for sp in ax3_kpis.spines.values():
        sp.set_visible(False)

    # KPI box outer frame fitted to the content
    ax3_kpis.add_patch(patches.FancyBboxPatch((0.02, 0.05), 0.96, 0.92, boxstyle="round,pad=0.02",
                                              facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.2))
    
    # KPI header
    ax3_kpis.add_patch(patches.FancyBboxPatch((0.05, 0.78), 0.90, 0.15, boxstyle="round,pad=0.01",
                                              facecolor="#ECFDF5", edgecolor="#10B981", linewidth=0.8))
    ax3_kpis.text(0.50, 0.855, "OPERATIONAL METRICS", fontsize=8.5, fontweight="bold", color="#047857", ha="center")

    # Internal grid of key metrics
    kpis_data = [
        ("Pipeline Latency", f"{latency_s:.2f} s", "#0F172A"),
        ("Sovereign Grid", "400,006 cells", "#0F172A"),
        ("Population Precision (Propagation)", f"{precision_prop_pob:.1f} %", "#15803D"),
        ("Chaco Spread ROC", f"{roc_prop_chaco:.3f}", "#DC2626"),
        ("Oriental Hybrid ROC", f"{roc_m3_oriental:.3f}", "#0284C7"),
        ("Chaco Hybrid ROC", f"{roc_m3_chaco:.3f}", "#0284C7"),
    ]
    y_kpi = 0.67
    for label_k, val_k, col_k in kpis_data:
        ax3_kpis.text(0.08, y_kpi, f"• {label_k}:", fontsize=6.8, color="#475569")
        ax3_kpis.text(0.92, y_kpi, val_k, fontsize=7.0, fontweight="bold", color=col_k, ha="right")
        y_kpi -= 0.11

    # Sub-ax 3C: International Benchmarking Bar Chart (Gran Chaco Blind Test)
    ax3_bench = fig.add_subplot(gs3[1, :])
    ax3_bench.set_facecolor("#FFFFFF")
    for sp in ax3_bench.spines.values():
        sp.set_color("#CBD5E1")

    modelos = ["Incendios_PY\n(Spread)", "Incendios_PY\n(Hybrid M3)", "Canadian\nFWI (ISI)", "Proxy INPE\nRisco Fogo"]
    roc_scores = [roc_prop_chaco, roc_m3_chaco, roc_fwi_chaco, roc_inpe_chaco]
    colores_bar = ["#DC2626", "#0284C7", "#64748B", "#94A3B8"]

    bars = ax3_bench.bar(range(len(modelos)), roc_scores, color=colores_bar, width=0.55, zorder=3)
    ax3_bench.axhline(0.5, color="#DC2626", linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)
    ax3_bench.set_ylim(0, 1.05)
    ax3_bench.set_xticks(range(len(modelos)))
    ax3_bench.set_xticklabels(modelos, fontsize=7.4, fontweight="bold")
    ax3_bench.set_ylabel("ROC-AUC (Gran Chaco)", fontsize=7.6, fontweight="bold", color="#0F172A")
    ax3_bench.grid(axis="y", color="#F1F5F9", linewidth=0.8, zorder=1)

    for bar, val in zip(bars, roc_scores):
        ax3_bench.text(bar.get_x() + bar.get_width()/2, val + 0.025, f"{val:.3f}", 
                       ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0F172A")

    ax3_bench.set_title("Gran Chaco Benchmarking on Blind Out-of-Time Test Set (2024–2025)", fontsize=8.2, fontweight="bold", color="#0F172A", pad=4)

    # -------------------------------------------------------------
    # STRICT HORIZONTAL FLOW ARROWS (SLOPE = 0)
    # Connect only in the inter-panel gutters
    # -------------------------------------------------------------
    # 1. Top Track (Physical Ignition): Pure horizontal at y = 0.72
    p1_to_p2_top = patches.ConnectionPatch(
        xyA=(1.0, 0.72), coordsA=ax1_bg.transAxes,
        xyB=(0.0, 0.72), coordsB=ax2_bg.transAxes,
        arrowstyle="-|>", mutation_scale=14, color="#0284C7", linewidth=2.0
    )
    p2_to_p3_top = patches.ConnectionPatch(
        xyA=(1.0, 0.72), coordsA=ax2_bg.transAxes,
        xyB=(0.0, 0.72), coordsB=ax3_bg.transAxes,
        arrowstyle="-|>", mutation_scale=14, color="#0284C7", linewidth=2.0
    )

    # 2. Bottom Track (Tactical Propagation): Pure horizontal at y = 0.26
    p1_to_p2_bot = patches.ConnectionPatch(
        xyA=(1.0, 0.26), coordsA=ax1_bg.transAxes,
        xyB=(0.0, 0.26), coordsB=ax2_bg.transAxes,
        arrowstyle="-|>", mutation_scale=14, color="#DC2626", linewidth=2.0
    )
    p2_to_p3_bot = patches.ConnectionPatch(
        xyA=(1.0, 0.26), coordsA=ax2_bg.transAxes,
        xyB=(0.0, 0.26), coordsB=ax3_bg.transAxes,
        arrowstyle="-|>", mutation_scale=14, color="#DC2626", linewidth=2.0
    )

    fig.patches.extend([p1_to_p2_top, p2_to_p3_top, p1_to_p2_bot, p2_to_p3_bot])

    # Explanatory labels over the horizontal arrows in the gutter
    # Channel 1 (between Panel 1 and Panel 2): x ≈ 0.329
    fig.text(0.3295, 0.735, "Physical\nDrivers", fontsize=6.8, fontweight="bold", color="#0284C7", ha="center")
    fig.text(0.3295, 0.275, "Thermal\nFoci (t-1)", fontsize=6.8, fontweight="bold", color="#DC2626", ha="center")

    # Channel 2 (between Panel 2 and Panel 3): x ≈ 0.670
    fig.text(0.6705, 0.735, "P(Ignition)\nForecast", fontsize=6.8, fontweight="bold", color="#0284C7", ha="center")
    fig.text(0.6705, 0.275, "P(Spread)\nValidation", fontsize=6.8, fontweight="bold", color="#DC2626", ha="center")

    # Save as PNG (300 DPI) and vector PDF
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    print(f"✓ Figura PNG (300 DPI) guardada en: {output_png}")

    if output_pdf:
        plt.savefig(output_pdf, format="pdf", facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        print(f"✓ Figura PDF vectorial guardada en: {output_pdf}")

    plt.close(fig)
    print("=" * 65)

if __name__ == "__main__":
    out_dir = ROOT_DIR / "entregables" / "figuras"
    png_path = out_dir / "fig01_arquitectura_impacto_nature.png"
    pdf_path = out_dir / "fig01_arquitectura_impacto_nature.pdf"
    crear_figura_impacto(png_path, pdf_path)
