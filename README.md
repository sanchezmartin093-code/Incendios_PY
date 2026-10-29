# Incendios_PY: a reproducible computational framework for decoupled wildfire ignition and propagation

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Data: VIIRS 375 m and ERA5-Land](https://img.shields.io/badge/Data-VIIRS%20375m%20%7C%20ERA5--Land-green.svg)](#data)
[![ML: LightGBM and Platt](https://img.shields.io/badge/ML-LightGBM%20%7C%20Platt-orange.svg)](#models)

`Incendios_PY` is a reproducible, open-source geospatial pipeline for daily wildfire
ignition and propagation modelling in Paraguay at 1 km² resolution. The repository
accompanies a manuscript prepared for *Computers & Geosciences*.

The framework separates two prediction problems:

- **Primary ignition:** fire at day `t` with no fire at `t-1` within 25 km and no
  same-cell fire in the previous 10 days (target D2).
- **Tactical propagation:** fire at day `t` within 25 km of a fire at `t-1`.

The operational layer uses the hybrid ignition model **M3** for cells with observed
fire history and the base model **M2** as a fallback for cells without history. **M1**
is a diagnostic biophysical baseline; the 2023 validation rejected it as the fallback.
Propagation is evaluated only in the `t-1` fire domain.

## Status

> **Documentation status.** Main results below are the Phase 2 selection and
> retrospective test. Model selection used 2023 validation only; 2024–2025 is
> retrospective and was not used for selection. See `auditoria/FASE2_DECISION_MODELO.md`.

- Data: `dataset_unificado_candidatos` annual-fuel build, 8 annual partitions,
  915,561,114 candidate cell-days, 867,430 fire cell-days, 397,448 cells.
  Two independent builds produced identical SHA-256 for all 8 partitions.
- VIIRS: 892,533 fire cell-days after the corrected `low`/`l` confidence filter.
- Climate: ERA5-Land CDS 3-hourly; 0-7 cm soil moisture from CDS and a
  Hargreaves-Samani reference-ET0 approximation; antecedent frost/water-deficit
  features. Training starts on `2018-03-02` after a 60-day warm-up.
- Fuel: MapBiomas Collection 2 annual maps 2017-2023, joined to each row with
  the prior-year map (`Y-1`); 2024 uses 2023 (`fuel_lag=1`) and 2025 reuses 2023
  (`fuel_lag=2`). Pasture, annual agriculture, forest plantation, savanna,
  woodlands, flooded grassland and forest-agriculture edge are separated.
- Loss: Hansen GFC-2025-v1.13 annual tree-cover loss at `t-1` and `t-2`.
- Targets: population-based D2 sampling at 1:10 per task and region.
- Holdout: 31 confirmation blocks (50 km, 20 km buffer), 68,719 cells; evaluated
  independently in 2023 after retraining without the confirmation blocks.
- Models: M1, M2, M3 and propagation boosters with monotonic Platt calibration.
- Alerts: M3 with M2 fallback; propagation restricted to the `t-1` domain.
- Tests: **76 pass** (pytest; 2 warnings).
- CI: exact version pins in `requirements-lock.txt`; wheel hashes are a documented
  exception (`auditoria/FASE2_0_LOCK_EXCEPTION.md`).
- Public release: **v2.1.0** (2026-10-28): <https://github.com/sanchezmartin093-code/Incendios_PY/releases/tag/v2.1.0>.
- Submission checklist: [`docs/CHECKLIST_ENVIO.md`](docs/CHECKLIST_ENVIO.md).

## Installation

Tested with Python 3.12.16 on macOS ARM64 and Linux x86_64.

```bash
git clone https://github.com/sanchezmartin093-code/Incendios_PY.git
cd Incendios_PY
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For an exact environment, use the pinned versions in `requirements-lock.txt`.
The lock does not include wheel hashes; see `auditoria/FASE2_0_LOCK_EXCEPTION.md`.

Core dependencies include Python 3.12+, Polars, DuckDB, PyArrow, GeoPandas,
Rasterio, Xarray, LightGBM, scikit-learn, SHAP, Matplotlib and CDS API. GDAL/GEOS
system libraries are required by GeoPandas and Rasterio.

## Quick start

Run the test suite:

```bash
python -m pytest -q tests
```

The full data pipeline requires the raw NASA FIRMS, ERA5-Land, MapBiomas, Hansen GFC and
administrative boundary inputs documented in `fuentes/MANIFIESTO_DATOS_v3.csv`.

See `docs/USER_GUIDE.md` for inputs, outputs, options and troubleshooting, and
`docs/TUTORIAL.md` for a minimal reproduction example.

The minimal operational artifact bundle (M3, M2 fallback, propagation,
calibrators, thresholds and D2 prevalences) can be rebuilt with:

```bash
python scripts/34_empaquetar_operativo.py
```

It is written to `dist/operativo_v2.1.0/` and deliberately excludes M1. M1 remains
in the scientific archive and in the supplementary material as the diagnostic
biophysical baseline.
Raw data and trained binaries are not versioned in Git.

Canonical pipeline:

```bash
# 1. Base data
python scripts/00_generar_grilla_nacional.py
python scripts/01b_descargar_firms_api.py
python scripts/01_descargar_firms.py
python scripts/02b_descargar_era5_cds.py --workers 2
python scripts/02f_completar_clima_cds.py        # short CDS path: swvl1 + Hargreaves ET0
# if annual CDS jobs are too heavy: python scripts/02f_completar_clima_cds.py --trimestre
# 02c is optional and requires Open-Meteo quota; not needed for the short CDS path
python scripts/02d_calcular_fwi_completo.py
python scripts/02e_calcular_inpe.py
python scripts/03_procesar_combustible.py
python scripts/03a_procesar_combustible_anual.py  # prior-year fuel (2017-2023 maps)
python scripts/03b_descargar_desmonte_hansen.py   # Hansen GFC 2025 v1.13
python scripts/04_construir_dataset.py

# 2. D2 targets, population weights and v3 feature injection
python scripts/05d_muestrear_poblacional_d2.py
python scripts/05e_inyectar_features_v3.py
python scripts/19_estimar_prevalencias.py

# 3. Models and evaluation
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
python scripts/24_entrenar_m1.py
python scripts/26_validacion_2023.py
python scripts/27_validacion_espacial_2023.py
python scripts/29_lead_time_2023.py
python scripts/30_validacion_propagacion_2023.py
python scripts/35_evaluar_politica_gated.py
python scripts/37_validacion_confirmacion_2023.py
python scripts/38_coupled_baseline.py
python scripts/39_sensibilidad_d2.py --reuse

# 4. Operational products
python scripts/09_servicio_alerta.py --fecha 2024-09-09
python scripts/10_generar_figuras_manuscrito.py
python scripts/10b_generar_figura_benchmarking.py
python scripts/11_exportar_geotiff.py --fecha 2024-09-09
python scripts/12_generar_dashboard_interactivo.py --fecha 2024-09-09
python scripts/15_generar_figura_arquitectura_impacto.py
```

## Main results

### 2023 selection

| Region | Model | ROC-AUC | PR-AUC | F0.5 |
|---|---|---:|---:|---:|
| Chaco | M1 biophysical | 0.6354 | 0.1480 | 0.1762 |
| Chaco | M2 base | 0.7817 | 0.2452 | 0.2933 |
| Chaco | M3 hybrid | 0.7844 | 0.2542 | 0.2979 |
| Oriental | M1 biophysical | 0.7041 | 0.1818 | 0.2301 |
| Oriental | M2 base | 0.7496 | 0.2221 | 0.2631 |
| Oriental | M3 hybrid | 0.7628 | 0.2441 | 0.2875 |

M3 was retained for temporal nowcasting in both regions (paired ΔROC-AUC > 0 and
lower 95 % CI above −0.01). In 2023 spatial blocks, M2 was the most robust model
(0.6773/0.7470 versus 0.6496/0.7277 for M3 and 0.6456/0.7176 for M1). On cells
without history, M1 reached 0.6650/0.7018 versus 0.7677/0.7440 for M2 and
0.7626/0.7421 for M3, so M1 was rejected as the operational fallback.

### Retrospective test (2024–2025)

| Region | Model | ROC-AUC | PR-AUC | Case-control F0.5 precision |
|---|---|---:|---:|---:|
| Chaco | M3 hybrid ignition | 0.8021 | 0.2842 | 32.82 % |
| Chaco | M2 ignition baseline | 0.7991 | 0.2766 | 33.28 % |
| Chaco | Tactical propagation | 0.9039 | 0.6935 | 83.69 % |
| Oriental | M3 hybrid ignition | 0.7534 | 0.2340 | 30.13 % |
| Oriental | M2 ignition baseline | 0.7406 | 0.2146 | 26.09 % |
| Oriental | Tactical propagation | 0.7859 | 0.3752 | 65.03 % |

The case-control test set has a 1:10 sampling ratio (approximately 9.09 % prevalence).
These are **not** operational population-level precision values. The test was already
observed during development and is reported as retrospective, not blind. 2024 uses
the 2023 fuel map with lag 1; 2025 uses 2023 with lag 2.

### Population-level evaluation

| Region | Model | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| Chaco | M3 ignition | 0.082 % | 27.30 % | 0.8059 | 0.0009 |
| Chaco | M2 ignition | 0.090 % | 24.59 % | 0.8078 | 0.0007 |
| Chaco | Propagation | 9.67 % | 50.65 % | 0.9088 | 0.1270 |
| Oriental | M3 ignition | 0.198 % | 20.73 % | 0.7627 | 0.0019 |
| Oriental | M2 ignition | 0.166 % | 22.11 % | 0.7549 | 0.0015 |
| Oriental | Propagation | 4.78 % | 18.77 % | 0.7963 | 0.0287 |

Population-level ignition precision is low; the system is a nowcast and
prioritisation tool, not a stand-alone early-warning system.

### Benchmarking and robustness

- Full Canadian FWI: ROC-AUC 0.5059/0.5611 for ignition in Chaco/Oriental.
- INPE Risco de Fogo: 0.5266/0.4973.
- Grouped-by-date paired bootstrap: all main comparisons against FWI/INPE have
  `p ≤ 0.001` (`B = 1,000`).
- Spatial block holdout 2023 (50 km blocks, 20 km buffer): M1 0.6456/0.7176,
  M2 0.6773/0.7470, M3 0.6496/0.7277. M2 is the spatial comparator; M3 is not a
  spatial extrapolator.
- Weather at `t-1` (2023): M3 0.7831/0.7613; M2 0.7818/0.7463; M1 0.6342/0.6965.
- Full pipeline: 1.569 s and 2,182.1 MB peak RAM on ARM64.
- Conformal empirical coverage: 89.85 %/90.19 % for nominal 90 %; 95.02 %/94.56 %
  for nominal 95 %, without a formal guarantee under temporal shift.

## Repository structure

```text
Incendios_PY/
├── scripts/                 # pipeline 00–24, including 05b/05c/05d and 10b
├── src/                     # ingestion, features, targets, models, metrics, alerts
├── tests/                   # 76 unit tests
├── datos/
│   ├── raw/                 # FIRMS, ERA5, MapBiomas, vectors (not versioned)
│   ├── processed/           # grid, climate, hotspots, dataset, targets
│   └── outputs/             # alerts, bulletins, GeoTIFF
├── modelos/                 # boosters and calibrators (not versioned)
├── docs/                    # user guide, tutorial, audit notes and protocol
├── manuscrito/              # Quarto sources (self-contained project)
│   ├── _quarto.yml
│   ├── assets/              # bibliography, CSL, DOCX reference
│   ├── en/                  # English manuscript and supplement
│   └── es/                  # Spanish manuscript and supplement
├── entregables/             # final DOCX/HTML and compiled figures/tables
│   ├── figuras/             # figure sources
│   ├── tablas/              # generated main and supplementary tables
│   ├── figuras_maestro.pdf  # all final figures, one page per figure
│   └── tablas_maestro.xlsx  # all final tables, one sheet per table
└── fuentes/MANIFIESTO_DATOS_v3.csv
```

## Data

Raw datasets are public but are not redistributed in this repository:

1. **NASA FIRMS** VIIRS Suomi-NPP, NOAA-20 and NOAA-21 active-fire detections:
   <https://firms.modaps.eosdis.nasa.gov/>
2. **Copernicus Climate Data Store** ERA5-Land reanalysis:
   <https://cds.climate.copernicus.eu/>
3. **MapBiomas** Collection 2, 2023 land cover for Paraguay.
4. **Hansen Global Forest Change** GFC-2025-v1.13 annual tree-cover loss:
   <https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/>
5. **geoBoundaries** ADM0/ADM1 boundaries: <https://www.geoboundaries.org/>

The consolidated datasets, hashes and byte sizes are listed in
`fuentes/MANIFIESTO_DATOS_v3.csv`. Processed data and trained models are documented in this manifest, available in the v2.1.0 GitHub release, and indexed by the data concept DOI.

Code is released under the MIT License. Derived data are released under CC BY 4.0
where permitted; see `LICENSE-DATA`. Third-party raw data retain their original
licenses and terms.

## Code and data availability

- Source code: <https://github.com/sanchezmartin093-code/Incendios_PY>
- Release: **v2.1.0** (2026-10-28) — <https://github.com/sanchezmartin093-code/Incendios_PY/releases/tag/v2.1.0>
- Code concept DOI: [`10.5281/zenodo.23675104`](https://doi.org/10.5281/zenodo.23675104).
- Data concept DOI: [`10.5281/zenodo.23675278`](https://doi.org/10.5281/zenodo.23675278).
- License: MIT.
- Contact: Martín Sánchez Insfrán, `sanchez.martin093@gmail.com`.
- ORCID: <https://orcid.org/0000-0003-4138-9668>

## Known limitations

1. ERA5-Land is used on a 0.25° node grid; 3-hourly data may miss intra-day VPD and
   wind peaks.
2. MapBiomas Collection 2 ends in 2023. Rows from 2024 use the 2023 map (lag 1),
   and 2025 uses 2023 (lag 2). Collection 3 is not integrated yet.
   ``et0_mm`` in the short CDS path is Hargreaves-Samani, not FAO-56 reference
   ET0; it is an explicit approximation.
3. D2 is a task-specific target definition; it is not a universal ignition model.
4. Spatial generalisation in unseen blocks is limited for all ignition models; M2
   is the most robust spatial comparator. M3 should not be presented as a spatial
   extrapolation model, and M1 is not the operational fallback.
5. The operational alert still requires institutional validation and a day-ahead
   forecast layer. The 2024–2025 test is retrospective, and about 90.65 % of test
   cells overlap training cells through the temporal split.
6. The model is a nowcast/hindcast: weather features are same-day.
7. Ignition population precision is very low; operational use must be prioritised
   and evaluated prospectively.
8. A coupled t-1 comparator and a full 15/25/35-km × 7/10/14-d D2 sensitivity are reported in Tables S19–S20; external-country validation is not included yet.

## Tests

```bash
python -m pytest -q tests
```

The suite covers target definitions, climate aggregation, VIIRS filtering, monotonic
calibration, metrics, bootstrap resampling and the alert layer.

## Citation

See `CITATION.cff`. Cite the v2.1.0 release (2026-10-28) and the concept DOIs.

## License

Code: MIT. See `LICENSE`.
