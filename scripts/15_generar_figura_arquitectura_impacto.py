"""
Script 15: Generador Automatizado de Figura Científica de Alto Impacto (Nivel Nature / Elsevier Q1).
- Corrección de problemas de maquetación:
  1. Cuadro de MapBiomas: dimensión calculada con padding inferior (y >= 0.03) para evitar cortes en el borde.
  2. Cuadro de Métricas Operacionales: rediseñado con proporciones balanceadas y tarjetas KPI estructuradas.
  3. Flechas de flujo: 100% HORIZONTALES (pendiente 0) conectadas en los canales inter-paneles sin cruzar ejes ni texto.
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
import geopandas as gpd
import polars as pl
import numpy as np

def crear_figura_impacto(
    output_png: Path,
    output_pdf: Path | None = None
):
    print("=" * 65)
    print("GENERANDO FIGURA CIENTÍFICA (LAYOUT CORREGIDO: FLECHAS HORIZONTALES Y SIN CORTES)")
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

    # 2. Configuración de Tipografía y Estilo Editorial
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.8

    # Lienzo panorámico de alta definición: 20 x 8.5 pulgadas (300 DPI)
    fig = plt.figure(figsize=(20, 8.5), facecolor="#FFFFFF", dpi=300)

    # -------------------------------------------------------------
    # CABECERA EDITORIAL GLOBAL
    # -------------------------------------------------------------
    fig.text(0.03, 0.965, "INCENDIOS_PY · SOVEREIGN GEOAI SYSTEM", fontsize=11, fontweight="bold", color="#1E40AF")
    fig.text(0.03, 0.932, "Decoupled Biophysical Ignition Hazard & Tactical Wildfire Propagation Framework", fontsize=16, fontweight="bold", color="#0F172A")
    fig.text(0.97, 0.938, "24-HOUR FORECAST · 1-km² RESOLUTION · SUB-SECOND INFERENCE", fontsize=10, fontweight="bold", color="#64748B", ha="right")

    # Grilla principal de 3 paneles horizontales
    # Gutter ancho entre paneles para dar espacio a flechas y etiquetas horizontales limpias
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

    # Banner superior de fase 1
    ax1_bg.text(0.04, 0.96, "PHASE 1", fontsize=8.5, fontweight="bold", color="#FFFFFF", 
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1E293B", edgecolor="none"))
    ax1_bg.text(0.19, 0.96, "Multisource Ingestion & Two-Stage Gating", fontsize=9.8, fontweight="bold", color="#0F172A", va="center")

    # Sub-grilla interna de Panel 1:
    # Fila 0: Mapa a la izquierda (43%) y 3 tarjetas a la derecha (57%)
    # Fila 1: Gating biofísico horizontal en la base
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[0, 0], 
                                          height_ratios=[0.63, 0.37], width_ratios=[0.43, 0.57], 
                                          wspace=0.10, hspace=0.12)

    # Sub-ax 1A: Mapa de Paraguay
    ax1_map = fig.add_subplot(gs1[0, 0])
    ax1_map.set_facecolor("none")
    adm0.boundary.plot(ax=ax1_map, color="#0F172A", linewidth=1.2, zorder=3)
    adm1.boundary.plot(ax=ax1_map, color="#94A3B8", linewidth=0.5, linestyle=":", zorder=2)
    
    # Límites estrictos para confinar el mapa a su columna
    ax1_map.set_xlim(-63.0, -53.8)
    ax1_map.set_ylim(-28.0, -18.8)
    ax1_map.set_xticks([])
    ax1_map.set_yticks([])
    for sp in ax1_map.spines.values():
        sp.set_visible(False)

    if df_test is not None:
        # Malla continua representativa de VPD en fondo
        vpd_pts = df_test.select(["lon", "lat", "vpd_max_kpa"]).to_pandas()
        ax1_map.scatter(vpd_pts["lon"], vpd_pts["lat"], c=vpd_pts["vpd_max_kpa"], 
                        cmap="YlOrBr", s=7, alpha=0.35, zorder=1)
        
        # Puntos de focos reales VIIRS en rojo
        focos = df_test.filter(pl.col("fuego") == 1)
        ax1_map.scatter(focos["lon"].to_numpy(), focos["lat"].to_numpy(), 
                        s=16, color="#DC2626", alpha=0.90, edgecolors="#7F1D1D", linewidth=0.5, zorder=4)

    ax1_map.set_title("Paraguay (1 km² Grid)\nVIIRS Fire Hotspots", fontsize=8.0, fontweight="bold", color="#0F172A", pad=2)

    # Sub-ax 1B: Tres Tarjetas de Fuentes Satelitales (Distribuídas equitativamente sin cortes)
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
        ("MapBiomas Collection 4", "30m aggregated to 1 km²\n5 dominant fuel/forest classes", "#16A34A", "#F0FDF4"),
    ]
    
    # 3 tarjetas con altura 0.27 y espaciado de 0.05 (desde y=0.04 hasta y=0.95 -> 0% de corte inferior)
    card_height = 0.26
    card_bottoms = [0.68, 0.36, 0.04]
    for (title_f, desc_f, col_f, bg_f), y_b in zip(fuentes_info, card_bottoms):
        ax1_cards.add_patch(patches.FancyBboxPatch((0.02, y_b), 0.96, card_height, boxstyle="round,pad=0.02",
                                                   facecolor=bg_f, edgecolor=col_f, linewidth=1.1))
        ax1_cards.text(0.06, y_b + card_height - 0.07, title_f, fontsize=8.2, fontweight="bold", color=col_f)
        ax1_cards.text(0.06, y_b + card_height - 0.20, desc_f, fontsize=7.2, color="#334155", linespacing=1.25)

    # Sub-ax 1C: Embudo de Gating Biofísico de Etapa 1
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
    ax1_gate.text(0.05, 0.18, "• Automatically excludes water, wet savannas & saturated air cells.\n• Discards 92% of non-fire national territory with 0% false dismissal.\n• Reduces computational space from 1,168.8M to candidate cell-days.", 
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

    # Banner superior de fase 2
    ax2_bg.text(0.04, 0.96, "PHASE 2", fontsize=8.5, fontweight="bold", color="#FFFFFF", 
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1D4ED8", edgecolor="none"))
    ax2_bg.text(0.19, 0.96, "Decoupled Dual-Layer GeoAI Engine", fontsize=9.8, fontweight="bold", color="#0F172A", va="center")

    # Tarjeta Superior: MODO 1 - IGNICIÓN FÍSICA (Centrada en y = 0.72)
    ax2_bg.add_patch(patches.FancyBboxPatch((0.04, 0.53), 0.92, 0.38, boxstyle="round,pad=0.02",
                                            facecolor="#F0F9FF", edgecolor="#0284C7", linewidth=1.4))
    
    # Tag de aislamiento causal
    ax2_bg.add_patch(patches.FancyBboxPatch((0.06, 0.84), 0.88, 0.05, boxstyle="round,pad=0.01",
                                            facecolor="#0284C7", edgecolor="none"))
    ax2_bg.text(0.50, 0.865, "MODE 1: PURE BIOPHYSICAL IGNITION HAZARD", fontsize=9.2, fontweight="bold", color="#FFFFFF", ha="center")
    
    ax2_bg.text(0.07, 0.77, "Features: Weather + Fuel Structure + Ecotone Edge Contrast", fontsize=8.5, fontweight="bold", color="#0F172A")
    ax2_bg.text(0.07, 0.69, "• STRICT CAUSAL ISOLATION: No fire history lags allowed (t-1 = 0).\n• Drivers: VPD max, Thermal amplitude, Wind speed, FFMC, Ecotone.\n• Calibrated via Monotonic Bounded Platt Scaling (w in [0.1, 10.0]).", 
                fontsize=7.6, color="#334155", linespacing=1.3)
    
    # Badge de salida Modo 1 (Alineado en y = 0.595)
    ax2_bg.add_patch(patches.FancyBboxPatch((0.07, 0.55), 0.86, 0.09, boxstyle="round,pad=0.015",
                                            facecolor="#E0F2FE", edgecolor="#0284C7", linewidth=0.8))
    ax2_bg.text(0.50, 0.595, "Output: P(Ignition)  -->  Pre-suppression & Forest Patrol", fontsize=8.4, fontweight="bold", color="#0369A1", ha="center")

    # Tarjeta Inferior: MODO 2 - PROPAGACIÓN TÁCTICA (Centrada en y = 0.26)
    ax2_bg.add_patch(patches.FancyBboxPatch((0.04, 0.06), 0.92, 0.40, boxstyle="round,pad=0.02",
                                            facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=1.4))
    
    # Tag de propagación
    ax2_bg.add_patch(patches.FancyBboxPatch((0.06, 0.39), 0.88, 0.05, boxstyle="round,pad=0.01",
                                            facecolor="#DC2626", edgecolor="none"))
    ax2_bg.text(0.50, 0.415, "MODE 2: TACTICAL FIRE SPREAD & CONTAGION", fontsize=9.2, fontweight="bold", color="#FFFFFF", ha="center")
    
    ax2_bg.text(0.07, 0.32, "Features: Physical Drivers + Spatial Proximity Kernel (t-1)", fontsize=8.5, fontweight="bold", color="#0F172A")
    ax2_bg.text(0.07, 0.23, "• Causal spatial memory: dist_foco_t1_km, focos_25km_t1, frp_25km_t1.\n• Models convective advection & active flame front advancement.\n• Evaluated with 20-km Dead-Zone spatial buffer to prevent leakage.", 
                fontsize=7.6, color="#334155", linespacing=1.3)
    
    # Badge de salida Modo 2 (Alineado en y = 0.135)
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

    # Banner superior de fase 3
    ax3_bg.text(0.04, 0.96, "PHASE 3", fontsize=8.5, fontweight="bold", color="#FFFFFF", 
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#065F46", edgecolor="none"))
    ax3_bg.text(0.19, 0.96, "Sovereign Alert & Benchmark Validation", fontsize=9.8, fontweight="bold", color="#0F172A", va="center")

    # Sub-grilla interna de Panel 3:
    # Fila 0: Mapa a la izquierda (45%) y KPIs estructurados a la derecha (55%)
    # Fila 1: Benchmarking internacional en la base (completo)
    gs3 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[0, 2], 
                                          height_ratios=[0.56, 0.44], width_ratios=[0.45, 0.55], 
                                          wspace=0.10, hspace=0.14)

    # Sub-ax 3A: Mapa Nacional de Predicción
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
        p_comb = pts["probabilidad_propagacion"].fillna(pts["probabilidad_ignicion"])
        ax3_map.scatter(pts["lon"], pts["lat"], c=p_comb, cmap="inferno", s=8, alpha=0.85, zorder=4)

    ax3_map.set_title("Operational Alert Map\n(1 km² Resolution)", fontsize=8.0, fontweight="bold", color="#0F172A", pad=2)

    # Sub-ax 3B: KPIs Operacionales (Rediseñado proporcionalmente sin sobredimensión)
    ax3_kpis = fig.add_subplot(gs3[0, 1])
    ax3_kpis.set_facecolor("none")
    ax3_kpis.set_xlim(0, 1)
    ax3_kpis.set_ylim(0, 1)
    ax3_kpis.set_xticks([])
    ax3_kpis.set_yticks([])
    for sp in ax3_kpis.spines.values():
        sp.set_visible(False)

    # Marco exterior del cuadro KPI ajustado al contenido
    ax3_kpis.add_patch(patches.FancyBboxPatch((0.02, 0.05), 0.96, 0.92, boxstyle="round,pad=0.02",
                                              facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.2))
    
    # Cabecera KPI
    ax3_kpis.add_patch(patches.FancyBboxPatch((0.05, 0.78), 0.90, 0.15, boxstyle="round,pad=0.01",
                                              facecolor="#ECFDF5", edgecolor="#10B981", linewidth=0.8))
    ax3_kpis.text(0.50, 0.855, "OPERATIONAL METRICS", fontsize=8.5, fontweight="bold", color="#047857", ha="center")

    # Cuadrícula interna de 4 métricas clave
    kpis_data = [
        ("Inference Latency", "< 0.75 s", "#0F172A"),
        ("Sovereign Grid", "400,006 cells", "#0F172A"),
        ("Tactical Precision", "86.73 %", "#15803D"),
        ("Chaco Spread ROC", "0.874", "#DC2626"),
        ("Oriental Ignition ROC", "0.798", "#0284C7"),
    ]
    y_kpi = 0.65
    for label_k, val_k, col_k in kpis_data:
        ax3_kpis.text(0.08, y_kpi, f"• {label_k}:", fontsize=7.2, color="#475569")
        ax3_kpis.text(0.92, y_kpi, val_k, fontsize=7.4, fontweight="bold", color=col_k, ha="right")
        y_kpi -= 0.13

    # Sub-ax 3C: Gráfico de Barras de Benchmarking Internacional (Gran Chaco Test Ciego)
    ax3_bench = fig.add_subplot(gs3[1, :])
    ax3_bench.set_facecolor("#FFFFFF")
    for sp in ax3_bench.spines.values():
        sp.set_color("#CBD5E1")

    modelos = ["Incendios_PY\n(Spread)", "Incendios_PY\n(Ignition)", "Proxy INPE\nRisco Fogo", "Canadian\nFWI (ISI)"]
    roc_scores = [0.874, 0.722, 0.504, 0.518]
    colores_bar = ["#DC2626", "#0284C7", "#94A3B8", "#64748B"]

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
    # FLECHAS DE FLUJO HORIZONTALES ESTRICTAS (PENDIENTE = 0)
    # Conectan únicamente en los canales inter-paneles (gutters)
    # -------------------------------------------------------------
    # 1. Pista Superior (Ignición Física): Horizontal pura en y = 0.72
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

    # 2. Pista Inferior (Propagación Táctica): Horizontal pura en y = 0.26
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

    # Etiquetas explicativas sobre las flechas horizontales en el canal (gutter)
    # Canal 1 (entre Panel 1 y Panel 2): x ≈ 0.329
    fig.text(0.3295, 0.735, "Physical\nDrivers", fontsize=6.8, fontweight="bold", color="#0284C7", ha="center")
    fig.text(0.3295, 0.275, "Thermal\nFoci (t-1)", fontsize=6.8, fontweight="bold", color="#DC2626", ha="center")

    # Canal 2 (entre Panel 2 y Panel 3): x ≈ 0.670
    fig.text(0.6705, 0.735, "P(Ignition)\nForecast", fontsize=6.8, fontweight="bold", color="#0284C7", ha="center")
    fig.text(0.6705, 0.275, "P(Spread)\nValidation", fontsize=6.8, fontweight="bold", color="#DC2626", ha="center")

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
