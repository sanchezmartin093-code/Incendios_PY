"""
Script 08: Global and Local Interpretability Analysis with TreeSHAP (Lundberg et al., 2020).
- Computes exact SHAP values using LightGBM's TreeSHAP algorithm (C++).
- Determines the mean physical importance of variables for Chaco vs. Oriental Region.
- Generates editorial-quality vector figures (PDF and PNG at 300 DPI) in figuras/:
  * fig04_shap_importancia_chaco.png / .pdf
  * fig04_shap_importancia_oriental.png / .pdf
  * fig04_shap_comparativa_chaco_vs_oriental.png / .pdf
- Exports formal contribution tables:
  * tablas/tabla03_shap_ranking_variables.csv
  * tablas/tabla03_shap_ranking_variables.md
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
import matplotlib.pyplot as plt
import lightgbm as lgb

from src.constantes import DIR_PROCESSED, DIR_MODELOS, DIR_FIGURAS, DIR_TABLAS
from src.modelo import preparar_matrices_xy

# Dictionary of readable names for scientific publication
NOMBRES_LEGIBLES = {
    "vpd_max_kpa": "Vapor pressure deficit (VPD, kPa)",
    "interaccion_viento_vpd": "Wind × VPD interaction",
    "rango_termico_diario": "Daily thermal range (°C)",
    "deficit_hidrico_acumulado": "Accumulated water deficit (VPD × CDD)",
    "potencial_viento_cuadratico": "Dynamic wind pressure (v²)",
    "viento_max_kmh": "Maximum wind speed (km/h)",
    "rafaga_max_kmh": "Maximum gust (km/h)",
    "ffmc": "Fine fuel moisture code (FFMC)",
    "isi": "Initial spread index (ISI)",
    "temp_max_c": "Maximum temperature (°C)",
    "temp_min_c": "Minimum temperature (°C)",
    "dias_secos_consecutivos": "Consecutive dry days (CDD)",
    "precipitacion_mm": "Daily rainfall (mm)",
    "precipitacion_48h_mm": "48-h accumulated rainfall (mm)",
    "pct_pastura": "Pasture/agriculture total (%)",
    "pct_pastura_ganadera": "Pasture cover (%)",
    "pct_agricultura_anual": "Annual agriculture cover (%)",
    "pct_plantacion_forestal": "Forest plantation cover (%)",
    "pct_sabana_pastizal": "Savanna / natural grassland cover (%)",
    "pct_bosque_seco": "Dry Chaco forest (%)",
    "pct_bosque_humedo": "Atlantic forest BAAPA (%)",
    "pct_humedal_palmar": "Flooded grassland / estero cover (%)",
    "borde_bosque_agro_pct": "Forest–agriculture edge fraction (%)",
    "swvl1_m3m3": "Topsoil moisture 0–7 cm (m³/m³)",
    "fdd_heladas_60d": "Antecedent frost severity, 60 days (°C-days)",
    "deficit_hidro_60d": "Antecedent water deficit, 60 days (mm)",
    "riesgo_estero_seco": "Dry estero risk (wetland × deficit)",
    "combustible_curado_helada": "Frost-cured pasture/grass fuel",
    "pct_desmonte_t1": "Tree-cover loss, Y-1 (%)",
    "pct_desmonte_t2": "Tree-cover loss, Y-2 (%)",
    "pct_desmonte_2a": "Tree-cover loss, Y-1 + Y-2 (%)",
    "presion_desmonte_vpd": "Deforestation × VPD pressure",
    "es_fin_de_semana": "Weekend effect (anthropogenic)",
    "dia_semana": "Day of week",
    "mes": "Month",
    "dia_del_anio": "Day of year",
    "indice_ecotono": "Forest–pasture ecotone fragmentation index",
    "dist_foco_t1_km": "Distance to t-1 fire (km)",
    "focos_25km_t1": "t-1 fire density (25-km radius)",
    "frp_25km_t1": "t-1 radiative power (FRP within 25 km)",
    "dist_foco_hist_km": "Distance to historical fire (km)",
    "densidad_hist_25km": "Historical ignition density (25 km)",
    "dist_corredor_antropico_km": "Distance to active anthropogenic corridor (km)",
    "focalizacion_antropica_local": "Local focalization index (5 km / 25 km)",
    "densidad_hist_5km": "Local anthropogenic ignition density (5 km)",
    "densidad_hist_10km": "Local anthropogenic ignition density (10 km)",
    "freq_hist_1km": "Chronic recurrence in a 1-km² cell",
    "presion_quema_pastura": "Pasture burning pressure (pasture × VPD × month)",
    "interaccion_acceso_vpd": "Accessibility–desiccation synergy"
}

def ejecutar_analisis_shap():
    print("=" * 65)
    print("ANÁLISIS DE INTERPRETABILIDAD CON TREESHAP (XAI)")
    print("=" * 65)

    p_val = DIR_PROCESSED / "targets_poblacionales" / "val_ignicion.parquet"
    if not p_val.exists():
        raise FileNotFoundError(f"Dataset no encontrado en {p_val}")

    df_val = pl.read_parquet(p_val)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)

    regiones = ["Chaco", "Oriental"]
    rankings_totales = []
    datos_plot = {}

    for region in regiones:
        print(f"\n--- CALCULANDO SHAP PARA: {region.upper()} ---")
        nombre_lower = region.lower()
        ruta_hibrida = DIR_MODELOS / f"lightgbm_hibrido_{nombre_lower}.booster"
        ruta_base = DIR_MODELOS / f"lightgbm_{nombre_lower}.booster"
        ruta_b = ruta_hibrida if ruta_hibrida.exists() else ruta_base
        modo = "hibrido" if ruta_hibrida.exists() else "ignicion"

        if not ruta_b.exists():
            print(f"Modelo para {region} no encontrado en {ruta_b}")
            continue

        booster = lgb.Booster(model_file=str(ruta_b))
        X_val, y_val, feats = preparar_matrices_xy(df_val, region, modo=modo)

        if len(X_val) == 0:
            print(f"Sin muestras de validación para {region}")
            continue

        # 1. Compute native TreeSHAP values in C++
        # booster.predict(..., pred_contrib=True) returns an array of (N, P + 1)
        contribuciones = booster.predict(X_val, pred_contrib=True)
        shap_valores = contribuciones[:, :-1]  # Exclude the intercept term (bias)

        # 2. Mean global importance: mean of the absolute value |SHAP|
        importancia_media = np.mean(np.abs(shap_valores), axis=0)

        df_rank = pd.DataFrame({
            "region": region,
            "feature": feats,
            "nombre_legible": [NOMBRES_LEGIBLES.get(f, f) for f in feats],
            "shap_importancia_media": importancia_media
        }).sort_values(by="shap_importancia_media", ascending=False).reset_index(drop=True)

        df_rank["ranking"] = df_rank.index + 1
        rankings_totales.append(df_rank)
        datos_plot[region] = df_rank

        print(f"Top 5 variables más influyentes en {region}:")
        for idx, row in df_rank.head(5).iterrows():
            print(f" {row['ranking']}. {row['nombre_legible']}: {row['shap_importancia_media']:.4f}")

        # 3. Generate individual horizontal bar chart
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        top12 = df_rank.head(12).sort_values(by="shap_importancia_media", ascending=True)
        color = "#c0392b" if region == "Chaco" else "#27ae60"
        
        bars = ax.barh(top12["nombre_legible"], top12["shap_importancia_media"], color=color, alpha=0.85, edgecolor="black")
        ax.set_xlabel("Mean absolute SHAP impact", fontsize=11, fontweight="bold")
        ax.set_title(f"Feature importance (TreeSHAP) — {region}", fontsize=13, fontweight="bold", pad=12)
        
        # Add value labels at the end of each bar
        for bar in bars:
            w = bar.get_width()
            ax.annotate(f"{w:.3f}",
                        xy=(w, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=9)

        plt.tight_layout()
        p_png = DIR_FIGURAS / f"fig04_shap_importancia_{nombre_lower}.png"
        p_pdf = DIR_FIGURAS / f"fig04_shap_importancia_{nombre_lower}.pdf"
        fig.savefig(p_png, dpi=300)
        fig.savefig(p_pdf)
        plt.close(fig)
        print(f" ✓ Gráfico exportado: {p_png.name} y {p_pdf.name}")

    # 4. Generate combined comparative Chaco vs. Oriental figure
    if len(datos_plot) == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300, sharex=False)
        
        # Chaco
        d_chaco = datos_plot["Chaco"].head(10).sort_values(by="shap_importancia_media", ascending=True)
        ax1.barh(d_chaco["nombre_legible"], d_chaco["shap_importancia_media"], color="#c0392b", alpha=0.85, edgecolor="black")
        ax1.set_title("Chaco (dry regime and dry biomass)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Mean |SHAP|", fontsize=10, fontweight="bold")

        # Oriental
        d_oriental = datos_plot["Oriental"].head(10).sort_values(by="shap_importancia_media", ascending=True)
        ax2.barh(d_oriental["nombre_legible"], d_oriental["shap_importancia_media"], color="#27ae60", alpha=0.85, edgecolor="black")
        ax2.set_title("Oriental Region (anthropogenic regime and pastures)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Mean |SHAP|", fontsize=10, fontweight="bold")

        plt.suptitle("TreeSHAP comparison of fire drivers by ecoregion", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        p_comp_png = DIR_FIGURAS / "fig04_shap_comparativa_chaco_vs_oriental.png"
        p_comp_pdf = DIR_FIGURAS / "fig04_shap_comparativa_chaco_vs_oriental.pdf"
        fig.savefig(p_comp_png, dpi=300)
        fig.savefig(p_comp_pdf)
        plt.close(fig)
        print(f"\n✓ Figura comparativa regional exportada: {p_comp_png.name} y {p_comp_pdf.name}")

    # 5. Export consolidated table
    if rankings_totales:
        df_ranking_final = pd.concat(rankings_totales, ignore_index=True)
        p_csv = DIR_TABLAS / "tabla03_shap_ranking_variables.csv"
        p_md = DIR_TABLAS / "tabla03_shap_ranking_variables.md"
        df_ranking_final.to_csv(p_csv, index=False)

        md_str = "# Tabla 3. Ranking de Importancia de Variables por Ecorregión (TreeSHAP)\n\n"
        md_str += df_ranking_final.to_markdown(index=False) + "\n"
        p_md.write_text(md_str, encoding="utf-8")
        print(f"✓ Tabla CSV exportada: {p_csv.name}")
        print(f"✓ Tabla Markdown exportada: {p_md.name}")

    print("=" * 65)

if __name__ == "__main__":
    ejecutar_analisis_shap()
