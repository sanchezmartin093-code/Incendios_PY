"""
Script 16: Ablation Study and Training of the Multiscale Socio-Ecological Hybrid Model.
- Rigorously compares on the blind out-of-time test set (2024–2025, N = 1 841 147):
  1. M1 — Pure Biophysical Model (18 variables: ERA5-Land meteorology, FWI and MapBiomas biomass, no anthropic).
  2. M2 — Base Ignition Model with Macro Anchor (25 variables: M1 + basic calendar and 25 km prior).
  3. M3 — Multiscale Socio-Ecological Hybrid Model (35 variables: M2 + agropastoral burning pressure,
          seasonal harmonic encoding, human access corridors at 1/5/10 km and Access × VPD interaction).
  4. Canonical International Benchmarking Systems (Pure Canadian FWI, INPE Fire Risk Proxy, Climatology).
- Exports:
  * modelos/lightgbm_hibrido_chaco.booster and lightgbm_hibrido_oriental.booster
  * modelos/calibradores/calibrador_hibrido_*.joblib
  * tablas/tabla05_ablacion_hibrido_socioecologico.csv and .md
  * tablas/tabla06_shap_modelo_hibrido.csv and .md
  * figuras/fig07_ablacion_biofisico_vs_hibrido.png and .pdf (300 DPI)
"""

from pathlib import Path
import sys
import json

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
    confusion_matrix,
    fbeta_score,
    roc_curve,
    precision_recall_curve
)

from src.constantes import DIR_PROCESSED, DIR_MODELOS, DIR_TABLAS, DIR_FIGURAS
from src.espaciotemporal import agregar_features_antropicas
from src.inpe import constante_a_desde_fracciones
from src.modelo import (
    entrenar_modelo_lightgbm,
    calibrar_probabilidades_isotonica,
    predecir_probabilidad,
    guardar_modelo_y_calibrador,
    FEATURES_M1_BIOFISICO_BASE,
    FEATURES_IGNICION_CHACO,
    FEATURES_IGNICION_ORIENTAL,
)
from src.metricas import (
    calcular_umbral_optimo_youden,
    calcular_umbral_optimo_f_beta,
    calcular_expected_calibration_error,
    calcular_bootstrap_ci_roc
)

# 1. M1 is imported from src.modelo so the ablation cannot drift from the
# production feature contract.
FEATURES_ANTROPICO_MULTIESCALA = [
    "dist_corredor_antropico_km",
    "densidad_hist_5km",
    "densidad_hist_10km",
    "temporada_quema_ganadera",
    "doy_sin",
    "doy_cos",
    "presion_quema_pastura",
    "focalizacion_antropica_local",
    "interaccion_acceso_vpd"
]

NOMBRES_LEGIBLES_HIBRIDO = {
    "dist_foco_hist_km": "Distancia a Ignición Histórica (km)",
    "densidad_hist_25km": "Densidad Macro de Ignición (25 km)",
    "densidad_hist_10km": "Densidad Meso de Ignición Antrópica (10 km)",
    "densidad_hist_5km": "Densidad Local de Ignición Antrópica (5 km)",
    "freq_hist_1km": "Recurrencia Crónica en Celda (1 km²)",
    "dist_corredor_antropico_km": "Distancia a Corredor Antrópico Activo (km)",
    "focalizacion_antropica_local": "Índice de Focalización Local (5km / 25km)",
    "interaccion_acceso_vpd": "Sinergia Accesibilidad × VPD",
    "presion_quema_pastura": "Presión de Quema en Pasturas (Pastura × VPD × Mes)",
    "temporada_quema_ganadera": "Ventana de Quema Agroganadera (Ago–Oct)",
    "doy_sin": "Armónico Estacional Seno (DOY)",
    "doy_cos": "Armónico Estacional Coseno (DOY)",
    "potencial_viento_cuadratico": "Presión Dinámica del Viento (v²)",
    "ffmc": "Humedad Combustible Fino (FFMC)",
    "isi": "Índice Propagación Inicial (ISI)",
    "temp_max_c": "Temperatura Máxima (°C)",
    "temp_min_c": "Temperatura Mínima (°C)",
    "rafaga_max_kmh": "Ráfaga Máxima de Viento (km/h)",
    "viento_max_kmh": "Velocidad de Viento Máx (km/h)",
    "precipitacion_48h_mm": "Precipitación Acumulada 48h (mm)",
    "rango_termico_diario": "Amplitud Térmica Diaria (°C)",
    "deficit_hidrico_acumulado": "Déficit Hídrico Acumulado (VPD × CDD)",
    "pct_pastura": "Fracción de Pastura Cultivada (%)",
    "pct_bosque_seco": "Fracción de Bosque Seco Chaqueño (%)",
    "pct_bosque_humedo": "Fracción de Bosque Atlántico (%)",
    "dia_semana": "Día de la Semana",
    "dia_del_anio": "Día Ordinal del Año (DOY)",
    "pct_humedal_palmar": "Fracción de Palmar/Humedal (%)",
    "precipitacion_mm": "Precipitación Diaria (mm)"
}

def construir_features_socioecologicas(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
    df_grilla: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Adds socio-ecological derived variables from the causal priors
    already computed in ``scripts/05_muestrear_ausencias.py``.

    The priors are not recomputed here: doing so with the training fires would
    introduce the event itself and future fires into the training features.
    """
    return (
        agregar_features_antropicas(df_train),
        agregar_features_antropicas(df_val),
        agregar_features_antropicas(df_test),
    )

def ejecutar_estudio_ablacion():
    print("=" * 75)
    print("ESTUDIO DE ABLACIÓN: BIOFÍSICO PURO vs. HÍBRIDO SOCIO-ECOLÓGICO vs. BENCHMARK")
    print("=" * 75)

    dir_targets = DIR_PROCESSED / "targets_poblacionales"
    df_train = pl.read_parquet(dir_targets / "train_ignicion.parquet")
    df_val = pl.read_parquet(dir_targets / "val_ignicion.parquet")
    df_test = pl.read_parquet(dir_targets / "test_ignicion.parquet")
    df_grilla = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet")

    print("1. Construyendo variables socio-ecológicas multiescala (Anti-Leakage)...")
    tr_h, va_h, te_h = construir_features_socioecologicas(df_train, df_val, df_test, df_grilla)

    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    dir_cal = DIR_MODELOS / "calibradores"
    dir_cal.mkdir(parents=True, exist_ok=True)

    filas_ablacion = []
    rankings_shap = []
    curvas_reg = {}

    for region in ["Chaco", "Oriental"]:
        print(f"\n--- PROCESANDO REGIÓN: {region.upper()} ---")
        tr_r = tr_h.filter(pl.col("region") == region)
        va_r = va_h.filter(pl.col("region") == region)
        te_r = te_h.filter(pl.col("region") == region)

        y_tr = tr_r["target_ignicion"].to_numpy().astype(np.float32)
        y_va = va_r["target_ignicion"].to_numpy().astype(np.float32)
        y_te = te_r["target_ignicion"].to_numpy().astype(int)

        p_clim = np.full(len(y_te), float(y_tr.mean()))
        bs_clim = float(brier_score_loss(y_te, p_clim))

        b_col = "pct_bosque_seco" if region == "Chaco" else "pct_bosque_humedo"

        # Identical capacity for M1, M2 and M3: the only difference between
        # architectures is the feature set. This way the ablation is not
        # confounded by more trees/leaves in M3.
        spw = 1.0 if region == "Chaco" else 2.0
        n_r_ctrl = 45 if region == "Chaco" else 60
        leaves_ctrl = 23 if region == "Chaco" else 31
        min_s = 20 if region == "Chaco" else 10

        # --- MODEL M1: PURE BIOPHYSICAL ---
        feats_m1 = FEATURES_M1_BIOFISICO_BASE + [b_col]
        X_tr_m1, X_va_m1, X_te_m1 = tr_r.select(feats_m1).to_numpy(), va_r.select(feats_m1).to_numpy(), te_r.select(feats_m1).to_numpy()
        b_m1 = entrenar_modelo_lightgbm(
            X_tr_m1, y_tr, X_va_m1, y_va, feats_m1,
            scale_pos_weight=spw, num_boost_round=n_r_ctrl,
            num_leaves=leaves_ctrl, min_child_samples=min_s,
            early_stopping_rounds=None,
        )
        c_m1 = calibrar_probabilidades_isotonica(b_m1, X_va_m1, y_va, metodo="platt_monotono")
        p_va_m1 = predecir_probabilidad(b_m1, c_m1, X_va_m1)
        p_te_m1 = predecir_probabilidad(b_m1, c_m1, X_te_m1)
        c_opt_m1, _ = calcular_umbral_optimo_f_beta(y_va, p_va_m1, beta=0.5)

        # --- MODEL M2: IGNITION WITH MACRO ANCHOR (retrained with the same capacity) ---
        feats_m2 = FEATURES_IGNICION_CHACO if region == "Chaco" else FEATURES_IGNICION_ORIENTAL
        X_tr_m2, X_va_m2, X_te_m2 = tr_r.select(feats_m2).to_numpy(), va_r.select(feats_m2).to_numpy(), te_r.select(feats_m2).to_numpy()
        b_m2 = entrenar_modelo_lightgbm(
            X_tr_m2, y_tr, X_va_m2, y_va, feats_m2,
            scale_pos_weight=spw, num_boost_round=n_r_ctrl,
            num_leaves=leaves_ctrl, min_child_samples=min_s,
            early_stopping_rounds=None,
        )
        c_m2 = calibrar_probabilidades_isotonica(b_m2, X_va_m2, y_va, metodo="platt_monotono")
        p_va_m2 = predecir_probabilidad(b_m2, c_m2, X_va_m2)
        p_te_m2 = predecir_probabilidad(b_m2, c_m2, X_te_m2)
        c_opt_m2, _ = calcular_umbral_optimo_f_beta(y_va, p_va_m2, beta=0.5)

        # --- MODEL M3: MULTISCALE SOCIO-ECOLOGICAL HYBRID ---
        feats_m3 = feats_m2 + FEATURES_ANTROPICO_MULTIESCALA + (["freq_hist_1km"] if region == "Oriental" else [])
        X_tr_m3, X_va_m3, X_te_m3 = tr_r.select(feats_m3).to_numpy(), va_r.select(feats_m3).to_numpy(), te_r.select(feats_m3).to_numpy()

        b_m3 = entrenar_modelo_lightgbm(
            X_tr_m3, y_tr, X_va_m3, y_va, feats_m3,
            scale_pos_weight=spw, num_boost_round=n_r_ctrl,
            num_leaves=leaves_ctrl, min_child_samples=min_s,
            early_stopping_rounds=None,
        )
        c_m3 = calibrar_probabilidades_isotonica(b_m3, X_va_m3, y_va, metodo="platt_monotono")
        p_va_m3 = predecir_probabilidad(b_m3, c_m3, X_va_m3)
        p_te_m3 = predecir_probabilidad(b_m3, c_m3, X_te_m3)
        c_opt_m3, max_f05_m3 = calcular_umbral_optimo_f_beta(y_va, p_va_m3, beta=0.5)
        c_youden_m3, max_j_m3 = calcular_umbral_optimo_youden(y_va, p_va_m3)

        # Save Serialized Hybrid Model and Validation Optimal Threshold Metadata (Anti-Leakage)
        guardar_modelo_y_calibrador(
            b_m3, c_m3,
            DIR_MODELOS / f"lightgbm_hibrido_{region.lower()}.booster",
            dir_cal / f"calibrador_hibrido_{region.lower()}.joblib"
        )
        meta_hibrido = {
            "region": region,
            "arquitectura": "M3_hibrido_socioecologico",
            "c_youden": round(float(c_youden_m3), 4),
            "c_f05": round(float(c_opt_m3), 4),
            "max_j_val": round(float(max_j_m3), 4),
            "max_f05_val": round(float(max_f05_m3), 4),
            "n_vars": len(feats_m3)
        }
        (dir_cal / f"umbrales_hibrido_{region.lower()}.json").write_text(
            json.dumps(meta_hibrido, indent=2), encoding="utf-8"
        )

        # Compute SHAP for Hybrid Model M3
        contrib_m3 = b_m3.predict(X_va_m3, pred_contrib=True)[:, :-1]
        imp_m3 = np.mean(np.abs(contrib_m3), axis=0)
        df_shap_reg = pd.DataFrame({
            "region": region,
            "feature": feats_m3,
            "nombre_legible": [NOMBRES_LEGIBLES_HIBRIDO.get(f, f) for f in feats_m3],
            "shap_media": imp_m3
        }).sort_values("shap_media", ascending=False).reset_index(drop=True)
        df_shap_reg["ranking"] = df_shap_reg.index + 1
        rankings_shap.append(df_shap_reg.head(10))

        # --- FULL INTERNATIONAL BENCHMARKS ---
        grilla = pl.read_parquet(
            DIR_PROCESSED / "grilla_nacional_1km.parquet",
            columns=["cell_id", "nodo_clima_id"],
        )
        fwi = pl.read_parquet(DIR_PROCESSED / "fwi_completo_diario.parquet")
        inpe = pl.read_parquet(DIR_PROCESSED / "inpe_diario.parquet")

        def _benchmark_scores(df):
            x = df.join(grilla, on="cell_id", how="left").join(
                fwi, on=["nodo_clima_id", "fecha"], how="left"
            ).join(inpe, on=["nodo_clima_id", "fecha"], how="left")
            a = np.array([
                constante_a_desde_fracciones(p, s, bs, bh, palm)
                for p, s, bs, bh, palm in zip(
                    x["pct_pastura"].to_numpy(),
                    x["pct_sabana_pastizal"].to_numpy(),
                    x["pct_bosque_seco"].to_numpy(),
                    x["pct_bosque_humedo"].to_numpy(),
                    x["pct_humedal_palmar"].to_numpy(),
                )
            ])
            madurez = np.clip(a * x["pse"].to_numpy() / 120.0, 0.0, 2.0)
            rb = 0.8 * (1 + np.sin(madurez * (np.pi / 2) - (np.pi / 2))) / 2
            fur = -0.006 * x["rh_min_pct"].to_numpy() + 1.3
            ft = 0.02 * x["temp_max_c"].to_numpy() + 0.4
            flat = 1 + np.abs(x["lat"].to_numpy()) * 0.003
            p_inpe = np.clip(rb * fur * ft * flat, 0, 1)
            return x["fwi"].to_numpy(), p_inpe

        p_fwi_val, p_inpe_val = _benchmark_scores(va_r)
        p_fwi, p_inpe = _benchmark_scores(te_r)
        c_opt_fwi, _ = calcular_umbral_optimo_f_beta(y_va.astype(int), p_fwi_val, beta=0.5)
        c_opt_inpe, _ = calcular_umbral_optimo_f_beta(y_va.astype(int), p_inpe_val, beta=0.5)

        evaluados = [
            ("M3: multiscale socio-ecological hybrid", len(feats_m3), p_te_m3, c_opt_m3, "#1f77b4", "-"),
            ("M2: macro-anchor ignition (v1)", len(feats_m2), p_te_m2, c_opt_m2, "#2ca02c", "--"),
            ("M1: biophysical only (no anthropic)", len(feats_m1), p_te_m1, c_opt_m1, "#ff7f0e", "-."),
            ("Benchmark: full Canadian FWI", 6, p_fwi, c_opt_fwi, "#9467bd", ":"),
            ("Benchmark: INPE Fire Risk", 7, p_inpe, c_opt_inpe, "#d62728", ":")
        ]

        curvas_reg[region] = {"y_true": y_te, "modelos": []}

        for nom, n_f, p_vec, cut, col, ls in evaluados:
            auc = float(roc_auc_score(y_te, p_vec))
            pr_auc = float(average_precision_score(y_te, p_vec))
            es_prob = nom.startswith("M") or "Híbrido" in nom
            if es_prob:
                bs = float(brier_score_loss(y_te, p_vec))
                bss = float(1.0 - (bs / (bs_clim + 1e-9)))
                ece = float(calcular_expected_calibration_error(y_te, p_vec))
            else:
                bs = bss = ece = float("nan")
            ci_l, ci_h = calcular_bootstrap_ci_roc(y_te, p_vec, n_bootstrap=300, block_ids=te_r["fecha"].to_numpy())

            y_bin = (p_vec >= cut).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_te, y_bin).ravel()
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f05 = float(fbeta_score(y_te, y_bin, beta=0.5, zero_division=0))

            filas_ablacion.append({
                "region": region,
                "arquitectura": nom,
                "n_vars": n_f,
                "roc_auc": round(auc, 4),
                "roc_ci_95": f"[{ci_l:.3f} - {ci_h:.3f}]",
                "pr_auc": round(pr_auc, 4),
                "brier_score": round(bs, 4) if np.isfinite(bs) else np.nan,
                "bss": round(bss, 4) if np.isfinite(bss) else np.nan,
                "ece": round(ece, 4) if np.isfinite(ece) else np.nan,
                "precision_pct": round(prec * 100, 2),
                "recall_pct": round(rec * 100, 2),
                "f05_score": round(f05, 4),
                "falsos_positivos": int(fp)
            })

            curvas_reg[region]["modelos"].append({
                "nombre": nom, "p": p_vec, "auc": auc, "pr_auc": pr_auc, "color": col, "ls": ls
            })
            print(f" • {nom:<38} | ROC={auc:.4f} | PR={pr_auc:.4f} | Prec={prec*100:.2f}% | FP={fp:,}")

    df_ab = pd.DataFrame(filas_ablacion)
    df_ab.to_csv(DIR_TABLAS / "tabla05_ablacion_hibrido_socioecologico.csv", index=False)
    (DIR_TABLAS / "tabla05_ablacion_hibrido_socioecologico.md").write_text(
        "# Tabla 5. Estudio de Ablación: Modelo Biofísico Puro vs. Híbrido Socio-Ecológico y Benchmarks\n\n" + df_ab.to_markdown(index=False) + "\n",
        encoding="utf-8"
    )

    df_sh = pd.concat(rankings_shap, ignore_index=True)
    df_sh.to_csv(DIR_TABLAS / "tabla06_shap_modelo_hibrido.csv", index=False)
    (DIR_TABLAS / "tabla06_shap_modelo_hibrido.md").write_text(
        "# Tabla 6. Top 10 Drivers en el Modelo Híbrido Socio-Ecológico (TreeSHAP)\n\n" + df_sh.to_markdown(index=False) + "\n",
        encoding="utf-8"
    )

    # Generate Ablation Figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    for i, reg in enumerate(["Oriental", "Chaco"]):
        y_true = curvas_reg[reg]["y_true"]
        ax_r, ax_p = axes[i, 0], axes[i, 1]
        for m in curvas_reg[reg]["modelos"]:
            fpr, tpr, _ = roc_curve(y_true, m["p"])
            prec, rec, _ = precision_recall_curve(y_true, m["p"])
            ax_r.plot(fpr, tpr, color=m["color"], linestyle=m["ls"], linewidth=2.0, label=f"{m['nombre']} ({m['auc']:.3f})")
            ax_p.plot(rec, prec, color=m["color"], linestyle=m["ls"], linewidth=2.0, label=f"{m['nombre']} ({m['pr_auc']:.3f})")
        ax_r.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Chance (0.500)")
        ax_r.set_title(f"Ablation ROC — {reg}", fontweight="bold")
        ax_r.set_xlabel("False positive rate (FPR)")
        ax_r.set_ylabel("True positive rate (TPR)")
        ax_r.legend(loc="lower right", fontsize=8)

        ax_p.axhline(float(y_true.mean()), color="grey", linestyle=":", label="Base prevalence")
        ax_p.set_title(f"Ablation precision–recall — {reg}", fontweight="bold")
        ax_p.set_xlabel("Recall")
        ax_p.set_ylabel("Operational precision (PPV)")
        ax_p.legend(loc="upper right", fontsize=8)

    plt.suptitle("Controlled ablation: biophysical M1 vs socio-ecological M3 vs benchmarks (retrospective test 2024–2025)", fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(DIR_FIGURAS / "fig07_ablacion_biofisico_vs_hibrido.png", dpi=300)
    fig.savefig(DIR_FIGURAS / "fig07_ablacion_biofisico_vs_hibrido.pdf")
    plt.close(fig)
    print("\n✓ Estudio de ablación finalizado y artefactos exportados.")

if __name__ == "__main__":
    ejecutar_estudio_ablacion()
