#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
R="${RSCRIPT:-Rscript}"
STAGE="${1:-all}"

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

run_py() { log "python $*"; "$PY" "$@"; }
run_r()  { log "Rscript $*"; "$R" "$@"; }

if [[ "$STAGE" == "data" || "$STAGE" == "all" ]]; then
  run_py scripts/00_generar_grilla_nacional.py
  run_py scripts/00_validar_datos.py
  run_py scripts/01_descargar_firms.py
  run_py scripts/02b_descargar_era5_cds.py
  run_py scripts/02d_calcular_fwi_completo.py
  run_py scripts/02e_calcular_inpe.py
  run_py scripts/03a_procesar_combustible_anual.py
  run_py scripts/04_construir_dataset.py
  run_py scripts/05d_muestrear_poblacional_d2.py
  run_py scripts/06_entrenar_modelos.py
  run_py scripts/07_evaluacion_test_2024.py
fi

if [[ "$STAGE" == "benchmark" || "$STAGE" == "all" ]]; then
  run_py scripts/13_benchmarking_modelos.py
  run_py scripts/18_evaluacion_poblacional.py
  run_py scripts/19_estimar_prevalencias.py
  run_py scripts/20_validacion_espacial_bloques.py
  run_py scripts/21_evaluacion_lead_time.py
  run_py scripts/43_benchmark_weather_only.py
  run_r  scripts/44_ajuste_multiplicidad.R
  run_py scripts/45_metricas_sin_solapamiento.py
  run_py scripts/46_bootstrap_pareado_m3m2.py
  run_py scripts/47_ic_poblacional_curvas.py
  run_py scripts/48_bootstrap_cost_loss.py
  run_py scripts/49_bootstrap_delta_valor.py
  run_py scripts/50_decision_curve_net_benefit.py
  run_py scripts/51_carga_operativa.py
  run_py scripts/52_sensibilidad_inpe_a_fija.py
  run_py scripts/53_rolling_origin.py
  run_py scripts/53b_figura_rolling.py
  run_py scripts/54_estabilidad_semillas.py
  run_py scripts/55_contexto_local_indices.py
  run_py scripts/56_sensibilidad_umbral.py
  run_py scripts/57_validacion_adversarial.py
  run_py scripts/60_control_negativo_permutacion.py
  run_py scripts/61_ablacion_gate.py
  run_py scripts/62_variantes_viirs.py
  run_py scripts/63_sensibilidad_muestreo.py
  run_py scripts/64_muestras_extra.py
  run_py scripts/65_nubes_era5.py
  run_py scripts/66_gate_sin_gate.py
  run_py scripts/67_muestreo_1_20_real.py
fi

if [[ "$STAGE" == "docs" || "$STAGE" == "all" ]]; then
  (cd manuscrito && quarto render)
  run_py scripts/40_formatear_docx_cageo.py entregables/en/manuscrito.docx entregables/manuscrito.docx
  run_py scripts/59_generar_hashes.py
fi

log "done: $STAGE"
