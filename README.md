# Incendios_PY: sistema predictivo de incendios en Paraguay

[![Licencia: MIT](https://img.shields.io/badge/C%C3%B3digo-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Datos: VIIRS 375m & ERA5-Land](https://img.shields.io/badge/Datos-VIIRS%20375m%20%7C%20ERA5--Land-green.svg)](#)
[![ML: LightGBM & Platt](https://img.shields.io/badge/ML-LightGBM%20%7C%20Platt-orange.svg)](#)

Sistema de modelado y alerta temprana de incendios para Paraguay a resolución de 1 km². La versión actual usa **targets D2**:

- **Ignición primaria:** fuego en t, sin fuego t-1 en 25 km y sin fuego en la misma celda en los últimos 10 días.
- **Propagación:** fuego en t dentro de 25 km de un fuego t-1.

## Estado actual

- Datos: **dataset_unificado v2**, 915.562.849 celdas-día, 867.354 fuegos, 397.336 celdas.
- Focos VIIRS: 892.533 celdas-día, filtro `l/low` corregido.
- Clima: ERA5-Land CDS **3-hourly**; precipitación diaria reconstruida; máximo 237,38 mm/día, 0 valores >300 mm.
- Combustible: MapBiomas real (`collection2_2023.tif`).
- Targets D2: muestreo poblacional 1:10 por target y dominio (`datos/processed/targets_poblacionales/`).
- Modelos: M2, M3 y propagación reentrenados; calibración Platt y umbrales F0.5 derivados en validación.
- Alertas: M3 en celdas con historia de fuego, M1 como fallback en celdas sin historia; propagación solo en dominio t-1; prevalencias D2.
- Tests: **62 OK**.

## Resultados en test D2

| Región | Modelo | ROC-AUC | PR-AUC | Precisión case-control F0.5 |
|---|---|---:|---:|---:|
| Chaco | M3 ignición híbrida | 0,7851 | 0,2410 | 27,54 % |
| Chaco | M2 ignición base | 0,7733 | 0,2148 | 23,10 % |
| Chaco | Propagación táctica | 0,8989 | 0,6831 | 84,94 % |
| Oriental | M3 ignición híbrida | 0,7485 | 0,2318 | 29,67 % |
| Oriental | M2 ignición base | 0,7324 | 0,2036 | 24,49 % |
| Oriental | Propagación táctica | 0,7799 | 0,3651 | 62,42 % |

La muestra case-control es 1:10 (prevalencia ~9,09 %). **No es precisión operativa.**

## Evaluación poblacional

| Región | Modelo | Precisión | Recall | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| Chaco | M3 ignición | 0,062 % | 27,56 % | 0,7884 | 0,0007 |
| Chaco | M2 ignición | 0,048 % | 38,07 % | 0,7771 | 0,0005 |
| Chaco | Propagación | 9,96 % | 48,91 % | 0,9050 | 0,1120 |
| Oriental | M3 ignición | 0,206 % | 22,78 % | 0,7618 | 0,0017 |
| Oriental | M2 ignición | 0,169 % | 24,55 % | 0,7507 | 0,0014 |
| Oriental | Propagación | 4,49 % | 19,61 % | 0,7913 | 0,0280 |

La prevalencia poblacional del dominio es ~0,015–0,041 % para ignición y ~0,155–0,210 % para propagación.

## Benchmarking y validación

- Canadian FWI completo: ROC-AUC 0,5185/0,5569 en ignición Chaco/Oriental.
- INPE Risco de Fogo: 0,5386/0,4994.
- Cost-loss: M3 no supera claramente a M2 en ignición; propagación muestra valor operativo cerca de α=0,002.
- Rendimiento: 1,569 s y 2.182,1 MB de RAM pico en ARM64.
- Bootstrap pareado agrupado por fecha: ver `tablas/tabla10_significancia_pareada.csv`.
- Holdout espacial: M1 0,6338/0,6176; M3 0,6243/0,5754. M3 **no** debe presentarse como modelo de extrapolación espacial.
- Clima t-1: M3 0,7841/0,7465; M1 0,6222/0,6340.
- Conformal: cobertura empírica 89,85/90,19 % para 90 %; 95,02/94,56 % para 95 %.

## Estructura

```text
Incendios_PY/
├── scripts/                 # pipeline 00–24 + 05b/05c/05d + 10b
├── src/                     # ingesta, rasgos, targets, modelos, métricas, alerta
├── tests/                   # 62 tests
├── datos/
│   ├── raw/                 # FIRMS, ERA5, MapBiomas, vectores (no versionado)
│   ├── processed/           # grilla, clima, focos, dataset, targets
│   └── outputs/             # alertas, boletines, GeoTIFF
├── modelos/                 # boosters y calibradores (no versionado)
├── tablas/                  # tablas 00–12 y S3–S5
├── figuras/                 # figuras del manuscrito
├── docs/ADR_AUDITORIA.md
├── fuentes/MANIFIESTO_DATOS_v2.csv
├── manuscrito.qmd
└── material_suplementario.qmd
```

## Pipeline canónico

```bash
# 1. Datos base
python scripts/00_generar_grilla_nacional.py
python scripts/01b_descargar_firms_api.py
python scripts/01_descargar_firms.py
python scripts/02b_descargar_era5_cds.py --workers 2
python scripts/02d_calcular_fwi_completo.py
python scripts/02e_calcular_inpe.py
python scripts/03_procesar_combustible.py
python scripts/04_construir_dataset.py

# 2. Muestreo D2
python scripts/05d_muestrear_poblacional_d2.py
python scripts/19_estimar_prevalencias.py

# 3. Modelos y evaluación
python scripts/06_entrenar_modelos.py
python scripts/16_ablacion_modelo_hibrido.py
python scripts/07_evaluacion_test_2024.py
python scripts/08_analisis_shap.py
python scripts/13_benchmarking_modelos.py
python scripts/18_evaluacion_poblacional.py
python scripts/20_validacion_espacial_bloques.py
python scripts/21_evaluacion_lead_time.py
python scripts/22_analisis_cost_loss.py
python scripts/23_benchmark_computacional.py

# 4. Productos
python scripts/09_servicio_alerta.py --fecha 2024-09-09
python scripts/10_generar_figuras_manuscrito.py
python scripts/10b_generar_figura_benchmarking.py
python scripts/11_exportar_geotiff.py --fecha 2024-09-09
python scripts/12_generar_dashboard_interactivo.py --fecha 2024-09-09
python scripts/15_generar_figura_arquitectura_impacto.py
```

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Limitaciones principales

1. **Resolución climática:** CDS 3-hourly por límite de costo; los máximos intradiarios de VPD/viento son aproximados.
2. **MapBiomas:** se usa `collection2_2023`; el manuscrito debe declarar esa colección.
3. **Targets:** D2 reduce el problema a dominios específicos; no es un modelo universal de ignición.
4. **Generalización espacial:** M3 cae en bloques no vistos; M1 es más robusto, pero ambos tienen limitaciones.
5. **Alertas:** el producto operativo aún requiere validación institucional y una capa de pronóstico day-ahead.
6. **Datos/modelos:** no versionados en git; usar `fuentes/MANIFIESTO_DATOS_v2.csv` y Zenodo para archivar.
7. **Manuscrito:** sincronizado con las tablas D2, el bootstrap agrupado y los modelos deterministas (2026-10-26).

## Reproducibilidad

- Verificar artefactos:

```bash
python scripts/00_verificar_manifiesto.py
```

- El estado actual está en el commit `feat(audit): corregir datos, targets D2, modelos, alertas y figuras`.
