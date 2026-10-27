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

- Data: `dataset_unificado` v2, 915,562,849 candidate cell-days, 867,354 fire
  cell-days, 397,336 cells.
- VIIRS: 892,533 fire cell-days after the corrected `low`/`l` confidence filter.
- Climate: ERA5-Land CDS 3-hourly; 0-7 cm soil moisture from CDS and a
  potential-evaporation ET0 proxy; antecedent frost/water-deficit features.
- Fuel: MapBiomas Collection 2 (2023), split into pasture, annual agriculture,
  forest plantation, savanna, woodlands and flooded grassland; forest-agriculture edge.
- Loss: Hansen GFC-2025-v1.13 annual tree-cover loss at `t-1` and `t-2`.
- Targets: population-based D2 sampling at 1:10 per task and region.
- Models: M1, M2, M3 and propagation boosters with monotonic Platt calibration.
- Alerts: M3 with M1 fallback; propagation restricted to the `t-1` domain.
- Tests: **65 pass + 1 conditional skip** (v3 contract).
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
python scripts/02f_completar_clima_cds.py        # short CDS path: swvl1 + potential evaporation
# if annual CDS jobs are too heavy: python scripts/02f_completar_clima_cds.py --trimestre
# 02c is optional and requires Open-Meteo quota; not needed for the short CDS path
python scripts/02d_calcular_fwi_completo.py
python scripts/02e_calcular_inpe.py
python scripts/03_procesar_combustible.py
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
| Chaco | M3 hybrid ignition | 0.7851 | 0.2410 | 27.54 % |
| Chaco | M2 ignition baseline | 0.7733 | 0.2148 | 23.10 % |
| Chaco | Tactical propagation | 0.8989 | 0.6831 | 84.94 % |
| Oriental | M3 hybrid ignition | 0.7485 | 0.2318 | 29.67 % |
| Oriental | M2 ignition baseline | 0.7324 | 0.2036 | 24.49 % |
| Oriental | Tactical propagation | 0.7799 | 0.3651 | 62.42 % |

The case-control test set has a 1:10 sampling ratio (approximately 9.09 % prevalence).
These are **not** operational population-level precision values.

### Population-level evaluation

| Region | Model | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| Chaco | M3 ignition | 0.062 % | 27.56 % | 0.7884 | 0.0007 |
| Chaco | M2 ignition | 0.048 % | 38.07 % | 0.7771 | 0.0005 |
| Chaco | Propagation | 9.96 % | 48.91 % | 0.9050 | 0.1120 |
| Oriental | M3 ignition | 0.206 % | 22.78 % | 0.7618 | 0.0017 |
| Oriental | M2 ignition | 0.169 % | 24.55 % | 0.7507 | 0.0014 |
| Oriental | Propagation | 4.49 % | 19.61 % | 0.7913 | 0.0280 |

The population prevalence is approximately 0.015–0.041 % for ignition and
0.155–0.210 % for propagation. The population-level ignition precision is low; the
system is a nowcast and prioritisation tool, not a stand-alone early-warning system.

### Benchmarking and robustness

- Full Canadian FWI: ROC-AUC 0.5185/0.5569 for ignition in Chaco/Oriental.
- INPE Risco de Fogo: 0.5386/0.4994.
- Grouped-by-date paired bootstrap: all main comparisons against FWI/INPE have
  `p ≤ 0.001` (`B = 1,000`).
- Spatial block holdout (50 km blocks, 20 km buffer): M1 0.6338/0.6176,
  M3 0.6243/0.5754. M3 should not be presented as a spatial extrapolation model.
- Weather at `t-1`: M3 0.7841/0.7465; M1 0.6222/0.6340.
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
2. MapBiomas fuel is a static 2023 layer; no annual land-cover series is used.
   The v3 feature contract (pasture/agriculture/plantation split, edge fraction,
   soil moisture, frost/deficit and Hansen loss) is implemented, but the annual
   prior-year MapBiomas rebuild remains the next methodological step to remove
   future fuel information from the 2018-2022 training rows.
   ``et0_mm`` in the short CDS path is ``potential_evaporation``, not FAO-56
   reference ET0; it is an explicit proxy and models must be retrained.
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
