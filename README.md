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
fire history and the biophysical model **M1** as a fallback for cells without history.
Propagation is evaluated only in the `t-1` fire domain.

## Status

- Data: `dataset_unificado` annual-fuel build, 915,561,114 candidate cell-days,
  867,430 fire cell-days, 397,448 cells.
- VIIRS: 892,533 fire cell-days after the corrected `low`/`l` confidence filter.
- Climate: ERA5-Land CDS 3-hourly; 0-7 cm soil moisture from CDS and a
  Hargreaves-Samani reference-ET0 approximation; antecedent frost/water-deficit
  features.
- Fuel: MapBiomas Collection 2 annual maps 2017-2023, joined to each row with
  the prior-year map (`Y-1`); 2025 reuses 2023 (`fuel_lag=2`). Pasture, annual
  agriculture, forest plantation, savanna, woodlands, flooded grassland and
  forest-agriculture edge are separated.
- Loss: Hansen GFC-2025-v1.13 annual tree-cover loss at `t-1` and `t-2`.
- Targets: population-based D2 sampling at 1:10 per task and region.
- Models: M1, M2, M3 and propagation boosters with monotonic Platt calibration.
- Alerts: M3 with M1 fallback; propagation restricted to the `t-1` domain.
- Tests: **66 pass** (v3 contract).
- Public release: [`v1.0.0`](https://github.com/sanchezmartin093-code/Incendios_PY/releases/tag/v1.0.0).

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

Core dependencies include Python 3.12+, Polars, DuckDB, PyArrow, GeoPandas,
Rasterio, Xarray, LightGBM, scikit-learn, SHAP, Matplotlib and CDS API. GDAL/GEOS
system libraries are required by GeoPandas and Rasterio.

## Quick start

Run the test suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The full data pipeline requires the raw NASA FIRMS, ERA5-Land, MapBiomas and
administrative boundary inputs documented in `fuentes/MANIFIESTO_DATOS_v2.csv`.
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

# 4. Operational products
python scripts/09_servicio_alerta.py --fecha 2024-09-09
python scripts/10_generar_figuras_manuscrito.py
python scripts/10b_generar_figura_benchmarking.py
python scripts/11_exportar_geotiff.py --fecha 2024-09-09
python scripts/12_generar_dashboard_interactivo.py --fecha 2024-09-09
python scripts/15_generar_figura_arquitectura_impacto.py
```

## Main results

### Held-out temporal test (2024–2025)

| Region | Model | ROC-AUC | PR-AUC | Case-control F0.5 precision |
|---|---|---:|---:|---:|
| Chaco | M3 hybrid ignition | 0.8021 | 0.2843 | 33.97 % |
| Chaco | M2 ignition baseline | 0.7945 | 0.2677 | 32.18 % |
| Chaco | Tactical propagation | 0.9032 | 0.6910 | 85.11 % |
| Oriental | M3 hybrid ignition | 0.7533 | 0.2338 | 29.15 % |
| Oriental | M2 ignition baseline | 0.7369 | 0.2102 | 26.95 % |
| Oriental | Tactical propagation | 0.7842 | 0.3710 | 63.25 % |

The case-control test set has a 1:10 sampling ratio (approximately 9.09 % prevalence).
These are **not** operational population-level precision values.

### Population-level evaluation

| Region | Model | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| Chaco | M3 ignition | 0.086 % | 24.93 % | 0.8060 | 0.0011 |
| Chaco | M2 ignition | 0.086 % | 25.70 % | 0.8021 | 0.0007 |
| Chaco | Propagation | 10.24 % | 48.77 % | 0.9089 | 0.1218 |
| Oriental | M3 ignition | 0.191 % | 23.15 % | 0.7632 | 0.0019 |
| Oriental | M2 ignition | 0.167 % | 18.83 % | 0.7479 | 0.0014 |
| Oriental | Propagation | 4.54 % | 19.22 % | 0.7938 | 0.0277 |

The population prevalence is approximately 0.015–0.041 % for ignition and
0.155–0.210 % for propagation. The population-level ignition precision is low; the
system is a nowcast and prioritisation tool, not a stand-alone early-warning system.

### Benchmarking and robustness

- Full Canadian FWI: ROC-AUC 0.5185/0.5569 for ignition in Chaco/Oriental.
- INPE Risco de Fogo: 0.5386/0.4994.
- Grouped-by-date paired bootstrap: all main comparisons against FWI/INPE have
  `p ≤ 0.001` (`B = 1,000`).
- Spatial block holdout (50 km blocks, 20 km buffer): M1 0.6544/0.6875,
  M3 0.6725/0.7010. M3 should not be presented as a spatial extrapolation model.
- Weather at `t-1`: M3 0.8014/0.7516; M1 0.6358/0.6794.
- Full pipeline: 1.569 s and 2,182.1 MB peak RAM on ARM64.
- Conformal empirical coverage: 89.85 %/90.19 % for nominal 90 %; 95.02 %/94.56 %
  for nominal 95 %, without a formal guarantee under temporal shift.

## Repository structure

```text
Incendios_PY/
├── scripts/                 # pipeline 00–24, including 05b/05c/05d and 10b
├── src/                     # ingestion, features, targets, models, metrics, alerts
├── tests/                   # 62 unit tests
├── datos/
│   ├── raw/                 # FIRMS, ERA5, MapBiomas, vectors (not versioned)
│   ├── processed/           # grid, climate, hotspots, dataset, targets
│   └── outputs/             # alerts, bulletins, GeoTIFF
├── modelos/                 # boosters and calibrators (not versioned)
├── tablas/                  # generated tables 00–12 and S3–S5
├── figuras/                 # manuscript figures
├── docs/                    # audit notes, benchmarking protocol and ADR
├── fuentes/MANIFIESTO_DATOS_v2.csv
├── manuscrito.qmd
└── material_suplementario.qmd
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
`fuentes/MANIFIESTO_DATOS_v2.csv`. Processed data and trained models are archived in Zenodo with DOI
[`10.5281/zenodo.23675279`](https://doi.org/10.5281/zenodo.23675279).

Code is released under the MIT License. Derived data are released under CC BY 4.0
where permitted; see `LICENSE-DATA`. Third-party raw data retain their original
licenses and terms.

## Code and data availability

- Source code: <https://github.com/sanchezmartin093-code/Incendios_PY>
- Release: <https://github.com/sanchezmartin093-code/Incendios_PY/releases/tag/v1.0.0>
- Code archive DOI: [`10.5281/zenodo.23675105`](https://doi.org/10.5281/zenodo.23675105) (concept DOI: `10.5281/zenodo.23675104`).
- Data archive DOI: [`10.5281/zenodo.23675279`](https://doi.org/10.5281/zenodo.23675279) (concept DOI: `10.5281/zenodo.23675278`).
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
4. M3 loses spatial generalisation in unseen blocks; M1 is more robust but also limited.
5. The operational alert still requires institutional validation and a day-ahead
   forecast layer.
6. The model is a nowcast/hindcast: weather features are same-day.
7. Ignition population precision is very low; operational use must be prioritised
   and evaluated prospectively.
8. No coupled `t-1` baseline or external-country validation is included yet.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The suite covers target definitions, climate aggregation, VIIRS filtering, monotonic
calibration, metrics, bootstrap resampling and the alert layer.

## Citation

See `CITATION.cff`. Cite the code archive as `10.5281/zenodo.23675105` and the data archive as `10.5281/zenodo.23675279`.

## License

Code: MIT. See `LICENSE`.
