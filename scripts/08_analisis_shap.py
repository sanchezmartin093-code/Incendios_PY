"""
Script 08: Análisis de Interpretabilidad Global y Local con TreeSHAP (Lundberg et al., 2020).
- Calcula valores SHAP exactos mediante el algoritmo TreeSHAP de LightGBM (C++).
- Determina la importancia física media de las variables para el Chaco vs. Región Oriental.
- Genera figuras vectoriales de calidad editorial (PDF y PNG a 300 DPI) en figuras/:
  * fig04_shap_importancia_chaco.png / .pdf
  * fig04_shap_importancia_oriental.png / .pdf
  * fig04_shap_comparativa_chaco_vs_oriental.png / .pdf
- Exporta tablas formales de contribución:
  * tablas/tabla03_shap_ranking_variables.csv
  * tablas/tabla03_shap_ranking_variables.md
"""

from pathlib import Path
import sys

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
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

# Diccionario de nombres legibles para publicación científica
NOMBRES_LEGIBLES = {
    "vpd_max_kpa": "Déficit de Presión de Vapor (VPD, kPa)",
    "interaccion_viento_vpd": "Interacción Viento × VPD",
    "rango_termico_diario": "Amplitud Térmica Diaria (°C)",
    "deficit_hidrico_acumulado": "Déficit Hídrico Acumulado (VPD × CDD)",
    "potencial_viento_cuadratico": "Presión Dinámica del Viento (v²)",
    "viento_max_kmh": "Velocidad de Viento Máx (km/h)",
    "rafaga_max_kmh": "Ráfaga Máxima de Viento (km/h)",
    "ffmc": "Índice Hojarasca Fina (FWI FFMC)",
    "isi": "Índice Propagación Inicial (FWI ISI)",
    "temp_max_c": "Temperatura Máxima (°C)",
    "temp_min_c": "Temperatura Mínima (°C)",
    "dias_secos_consecutivos": "Días Secos Consecutivos (CDD)",
    "precipitacion_mm": "Precipitación Diaria (mm)",
    "precipitacion_48h_mm": "Precipitación Acumulada 48h (mm)",
    "pct_pastura": "Cobertura de Pastura (%)",
    "pct_sabana_pastizal": "Cobertura de Sabana / Pastizal (%)",
    "pct_bosque_seco": "Bosque Xerofítico Chaqueño (%)",
    "pct_bosque_humedo": "Bosque Atlántico BAAPA (%)",
    "pct_humedal_palmar": "Palmares de Karanda'y (%)",
    "es_fin_de_semana": "Efecto Fin de Semana (Antrópico)",
    "dia_semana": "Día de la Semana",
    "mes": "Estacionalidad (Mes)",
    "dia_del_anio": "Día del Año"
}

def ejecutar_analisis_shap():
    print("=" * 65)
    print("ANÁLISIS DE INTERPRETABILIDAD CON TREESHAP (XAI)")
    print("=" * 65)

    p_val = DIR_PROCESSED / "val_dataset.parquet"
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
        ruta_b = DIR_MODELOS / f"lightgbm_{nombre_lower}.booster"

        if not ruta_b.exists():
            print(f"Modelo para {region} no encontrado en {ruta_b}")
            continue

        booster = lgb.Booster(model_file=str(ruta_b))
        X_val, y_val, feats = preparar_matrices_xy(df_val, region)

        if len(X_val) == 0:
            print(f"Sin muestras de validación para {region}")
            continue

        # 1. Calcular valores TreeSHAP nativos en C++
        # booster.predict(..., pred_contrib=True) retorna array de (N, P + 1)
        contribuciones = booster.predict(X_val, pred_contrib=True)
        shap_valores = contribuciones[:, :-1]  # Excluir el término independiente (bias)

        # 2. Importancia global media: promedio del valor absoluto |SHAP|
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

        # 3. Generar gráfico de barras horizontal individual
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        top12 = df_rank.head(12).sort_values(by="shap_importancia_media", ascending=True)
        color = "#c0392b" if region == "Chaco" else "#27ae60"
        
        bars = ax.barh(top12["nombre_legible"], top12["shap_importancia_media"], color=color, alpha=0.85, edgecolor="black")
        ax.set_xlabel("Impacto Medio Absoluto en la Predicción |SHAP|", fontsize=11, fontweight="bold")
        ax.set_title(f"Importancia de Variables (TreeSHAP) - {region.upper()}", fontsize=13, fontweight="bold", pad=12)
        
        # Añadir etiquetas de valor al final de cada barra
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

    # 4. Generar figura comparativa combinada Chaco vs. Oriental
    if len(datos_plot) == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300, sharex=False)
        
        # Chaco
        d_chaco = datos_plot["Chaco"].head(10).sort_values(by="shap_importancia_media", ascending=True)
        ax1.barh(d_chaco["nombre_legible"], d_chaco["shap_importancia_media"], color="#c0392b", alpha=0.85, edgecolor="black")
        ax1.set_title("Chaco (Régimen de Sequía y Biomasa Seca)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Impacto Medio |SHAP|", fontsize=10, fontweight="bold")

        # Oriental
        d_oriental = datos_plot["Oriental"].head(10).sort_values(by="shap_importancia_media", ascending=True)
        ax2.barh(d_oriental["nombre_legible"], d_oriental["shap_importancia_media"], color="#27ae60", alpha=0.85, edgecolor="black")
        ax2.set_title("Región Oriental (Régimen Antrópico y Pasturas)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Impacto Medio |SHAP|", fontsize=10, fontweight="bold")

        plt.suptitle("Comparativa de Drivers Biofísicos de Fuego por Ecorregión (TreeSHAP)", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        p_comp_png = DIR_FIGURAS / "fig04_shap_comparativa_chaco_vs_oriental.png"
        p_comp_pdf = DIR_FIGURAS / "fig04_shap_comparativa_chaco_vs_oriental.pdf"
        fig.savefig(p_comp_png, dpi=300)
        fig.savefig(p_comp_pdf)
        plt.close(fig)
        print(f"\n✓ Figura comparativa regional exportada: {p_comp_png.name} y {p_comp_pdf.name}")

    # 5. Exportar tabla consolidada
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
