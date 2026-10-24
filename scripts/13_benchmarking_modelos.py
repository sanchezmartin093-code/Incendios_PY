"""
Script 13: Benchmarking Cuantitativo Internacional de Modelos de Peligro de Incendios.
- Evalúa rigurosamente sobre el conjunto ciego de test (test_dataset.parquet):
  1. Incendios_PY (LightGBM Two-Stage + Calibración de Platt Monótona)
  2. Canadian FWI Puro (Índice ISI - Van Wagner, 1987 / Copernicus EFFIS)
  3. Proxy INPE Risco de Fogo (Setzer et al., 2019 / Adoptado por INFONA-MADES)
  4. Línea Base Climatológica (Prevalencia empírica)
- Genera:
  * tablas/tabla04_benchmarking_internacional.csv
  * tablas/tabla04_benchmarking_internacional.md
  * figuras/fig05_benchmarking_curvas_roc_pr.png y .pdf (300 DPI)
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
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    fbeta_score
)

from src.constantes import DIR_PROCESSED, DIR_MODELOS, DIR_TABLAS, DIR_FIGURAS
from src.modelo import cargar_modelo_y_calibrador, predecir_probabilidad, preparar_matrices_xy
from src.metricas import calcular_umbral_optimo_f_beta

def ejecutar_benchmarking():
    print("=" * 70)
    print("BENCHMARKING CUANTITATIVO: INCENDIOS_PY vs. FWI vs. INPE vs. CLIMATOLOGÍA")
    print("=" * 70)

    p_test = DIR_PROCESSED / "test_dataset.parquet"
    p_train = DIR_PROCESSED / "train_dataset.parquet"

    if not p_test.exists() or not p_train.exists():
        raise FileNotFoundError("Datasets no encontrados en datos/processed/")

    df_test = pl.read_parquet(p_test)
    df_train = pl.read_parquet(p_train)

    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    regiones = ["Oriental", "Chaco"]
    resultados = []
    datos_curvas = {}

    for region in regiones:
        print(f"\n--- EVALUANDO REGIONAL: {region.upper()} ---")
        df_reg_test = df_test.filter(pl.col("region") == region)
        df_reg_train = df_train.filter(pl.col("region") == region)

        y_true = df_reg_test["fuego"].to_numpy().astype(int)
        n_pos = int(y_true.sum())
        n_tot = len(y_true)
        prevalencia = n_pos / n_tot
        print(f" • Muestras evaluadas: {n_tot} (Positivos: {n_pos}, Prevalencia: {prevalencia:.2%})")

        # 1. Incendios_PY
        ruta_b = DIR_MODELOS / f"lightgbm_{region.lower()}.booster"
        ruta_c = DIR_MODELOS / "calibradores" / f"calibrador_{region.lower()}.joblib"
        booster, calibrador = cargar_modelo_y_calibrador(ruta_b, ruta_c)
        X_reg, _, _ = preparar_matrices_xy(df_test, region)
        p_incendios = predecir_probabilidad(booster, calibrador, X_reg)

        # 2. Canadian FWI (ISI normalizado 0 a 1)
        isi_raw = df_reg_test["isi"].to_numpy()
        p_fwi = (isi_raw - isi_raw.min()) / (isi_raw.max() - isi_raw.min() + 1e-6)

        # 3. Proxy INPE Risco de Fogo (Ecuación exponencial calibrada)
        veg = df_reg_test["pct_pastura"].to_numpy() if region == "Oriental" else df_reg_test["pct_bosque_seco"].to_numpy()
        cdd_vpd = df_reg_test["dias_secos_consecutivos"].to_numpy() * df_reg_test["vpd_max_kpa"].to_numpy() * (0.5 + veg)
        # Transformación logística estándar para obtener probabilidades P in [0, 1]
        p_inpe = 1.0 - np.exp(-0.05 * cdd_vpd)
        p_inpe = np.clip(p_inpe, 0.0, 1.0)

        # 4. Línea Base Climatológica (Promedio histórico de entrenamiento)
        p_clim = np.full(n_tot, float(df_reg_train["fuego"].mean()))
        bs_clim = brier_score_loss(y_true, p_clim)

        modelos = [
            ("Incendios_PY (GeoAI)", p_incendios, "#c0392b" if region == "Chaco" else "#27ae60", "-"),
            ("Proxy INPE Risco de Fogo", p_inpe, "#e67e22", "--"),
            ("Canadian FWI (ISI)", p_fwi, "#2980b9", "-."),
            ("Línea Base Climatológica", p_clim, "#7f8c8d", ":")
        ]

        datos_curvas[region] = {"y_true": y_true, "modelos": []}

        for nombre_mod, p_mod, color, ls in modelos:
            auc = float(roc_auc_score(y_true, p_mod)) if len(np.unique(p_mod)) > 1 else 0.5
            pr_auc = float(average_precision_score(y_true, p_mod)) if len(np.unique(p_mod)) > 1 else prevalencia
            bs = float(brier_score_loss(y_true, p_mod))
            bss = float(1.0 - (bs / (bs_clim + 1e-9)))

            # Punto de corte operativo F0.5
            if len(np.unique(p_mod)) > 1:
                c_opt, _ = calcular_umbral_optimo_f_beta(y_true, p_mod, beta=0.5)
                y_bin = (p_mod >= c_opt).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
                sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                f05 = float(fbeta_score(y_true, y_bin, beta=0.5, zero_division=0))
            else:
                c_opt, tp, fp, tn, fn = p_clim[0], 0, 0, n_tot - n_pos, n_pos
                sens, prec, f05 = 0.0, 0.0, 0.0

            resultados.append({
                "region": region,
                "modelo": nombre_mod,
                "roc_auc": round(auc, 4),
                "pr_auc": round(pr_auc, 4),
                "brier_score": round(bs, 4),
                "brier_skill_score": round(bss, 4),
                "cutoff_f05": round(c_opt, 4),
                "sensibilidad": round(sens, 4),
                "precision": round(prec, 4),
                "f05_score": round(f05, 4),
                "falsos_positivos": int(fp)
            })

            datos_curvas[region]["modelos"].append({
                "nombre": nombre_mod,
                "p": p_mod,
                "auc": auc,
                "pr_auc": pr_auc,
                "color": color,
                "ls": ls
            })

            print(f" • {nombre_mod:<26}: ROC={auc:.4f} | PR={pr_auc:.4f} | BSS={bss:+.4f} | FP={fp}")

    # 5. Exportar tablas formales
    df_res = pd.DataFrame(resultados)
    p_csv = DIR_TABLAS / "tabla04_benchmarking_internacional.csv"
    p_md = DIR_TABLAS / "tabla04_benchmarking_internacional.md"
    df_res.to_csv(p_csv, index=False)

    md_str = "# Tabla 4. Benchmarking Internacional de Modelos de Peligro de Incendios sobre Test Ciego 2024\n\n"
    md_str += df_res.to_markdown(index=False) + "\n"
    p_md.write_text(md_str, encoding="utf-8")
    print(f"\n✓ Tabla guardada en {p_csv.name} y {p_md.name}")

    # 6. Generar Figura de Benchmarking Internacional (ROC y PR para ambas ecorregiones)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    for i, reg in enumerate(["Oriental", "Chaco"]):
        y_true = datos_curvas[reg]["y_true"]
        ax_roc = axes[i, 0]
        ax_pr = axes[i, 1]

        for mod in datos_curvas[reg]["modelos"]:
            if mod["nombre"] == "Línea Base Climatológica":
                continue
            # Curva ROC
            fpr, tpr, _ = roc_curve(y_true, mod["p"])
            ax_roc.plot(fpr, tpr, color=mod["color"], linestyle=mod["ls"], linewidth=2.0,
                        label=f"{mod['nombre']} (AUC = {mod['auc']:.3f})")

            # Curva PR
            prec, rec, _ = precision_recall_curve(y_true, mod["p"])
            ax_pr.plot(rec, prec, color=mod["color"], linestyle=mod["ls"], linewidth=2.0,
                       label=f"{mod['nombre']} (PR-AUC = {mod['pr_auc']:.3f})")

        # Líneas de referencia
        ax_roc.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Azar (AUC = 0.500)")
        ax_roc.set_title(f"Curvas ROC - {reg.upper()}", fontsize=11, fontweight="bold")
        ax_roc.set_xlabel("1 - Especificidad (FPR)", fontsize=9)
        ax_roc.set_ylabel("Sensibilidad (TPR)", fontsize=9)
        ax_roc.legend(loc="lower right", fontsize=8.5)

        prevalencia_reg = float(y_true.mean())
        ax_pr.axhline(prevalencia_reg, color="grey", linestyle=":", label=f"Azar Climatológico ({prevalencia_reg:.2%})")
        ax_pr.set_title(f"Curvas Precision-Recall - {reg.upper()}", fontsize=11, fontweight="bold")
        ax_pr.set_xlabel("Sensibilidad (Recall)", fontsize=9)
        ax_pr.set_ylabel("Precisión (PPV)", fontsize=9)
        ax_pr.legend(loc="upper right", fontsize=8.5)

    plt.suptitle("Benchmarking Cuantitativo: Incendios_PY vs. Canadian FWI vs. INPE Risco de Fogo (Test Ciego 2024)",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    p_png = DIR_FIGURAS / "fig05_benchmarking_curvas_roc_pr.png"
    p_pdf = DIR_FIGURAS / "fig05_benchmarking_curvas_roc_pr.pdf"
    fig.savefig(p_png, dpi=300)
    fig.savefig(p_pdf)
    plt.close(fig)
    print(f"✓ Figuras de benchmarking exportadas: {p_png.name} y {p_pdf.name}")
    print("=" * 70)

if __name__ == "__main__":
    ejecutar_benchmarking()
