# Incendios_PY user guide

This guide describes the inputs, outputs, options and expected behaviour of the
public release. The scientific protocol and frozen results are documented in the
manuscript and supplementary material.

## 1. What the package does

`Incendios_PY` provides:

- the D2 target operator for primary ignition and tactical propagation;
- the biophysical gate and M3/M2 ignition inference policy;
- the causal `t-1` propagation layer;
- a reproducible evaluation harness (temporal split, spatial blocks, grouped
  bootstrap, lead time and cost-loss);
- table and figure generation for the manuscript and supplement.

M3 is used where `freq_hist_1km > 0`; M2 is the fallback where
`freq_hist_1km == 0`. M1 is a diagnostic biophysical baseline and is not used by
the operational alert layer.

## 2. Installation

```bash
git clone https://github.com/sanchezmartin093-code/Incendios_PY.git
cd Incendios_PY
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python -m pytest -q tests
```

The lock contains exact versions but no wheel hashes. This exception is recorded
in `auditoria/FASE2_0_LOCK_EXCEPTION.md`.

## 3. Inputs

Raw third-party data are not redistributed. They are documented, with SHA-256
hashes, in `fuentes/MANIFIESTO_DATOS_v3.csv`:

- NASA FIRMS VIIRS active-fire detections;
- ERA5-Land meteorology from the Copernicus Climate Data Store;
- annual MapBiomas Collection 2 fuel maps (2017–2023);
- Hansen GFC-2025 v1.13 tree-cover loss;
- geoBoundaries ADM0/ADM1 boundaries.

The v2.0.0 processed archive contains the targets, model probabilities,
benchmarks, models, calibrators, thresholds, prevalence metadata, tables and the
exact environment needed to regenerate the main tables without the raw archive.

## 4. Outputs

Main generated locations:

| Output | Location |
|---|---|
| Main tables | `tablas/` |
| Figures | `figuras/` |
| Alert layers and bulletins | `datos/outputs/` |
| Frozen candidate checks | `auditoria/FASE5_CANDIDATO_*` |
| Manuscript PDF/DOCX/HTML | repository root |
| Supplementary PDF/DOCX/HTML | repository root |
| Operational artifact bundle | `dist/operativo_v2.0.0/` |

## 5. Typical use cases

### 5.1 Reproduce an evaluation table from the frozen archive

1. Download `Incendios_PY_data_v2.0.0.zip` from the v2.0.0 GitHub release.
2. Place its `datos/processed/` content next to the repository or point the
   scripts to that path.
3. Run, for example:

```bash
python scripts/07_evaluacion_test_2024.py
python scripts/26_validacion_2023.py
python scripts/35_evaluar_politica_gated.py
python scripts/37_validacion_confirmacion_2023.py
python scripts/38_coupled_baseline.py
```

The scripts fail closed if required validation metadata are missing; they do not
recompute test thresholds.

### 5.2 Run the daily alert for a date

```bash
python scripts/09_servicio_alerta.py --fecha 2024-09-09
```

Outputs are written to `datos/outputs/`:

- `raster_riesgo_diario/alerta_nacional_YYYY-MM-DD.parquet`
- `boletin_alerta_YYYY-MM-DD.md`
- `boletin_alerta_YYYY-MM-DD.json`

Expected behaviour: the alert is a nowcast/prioritisation layer, not a
stand-alone warning system. Population-level ignition precision is very low
(0.082 %/0.198 %), so the product must be interpreted with the manuscript's
calibration and prevalence assumptions.

### 5.3 Build the minimal operational bundle

```bash
python scripts/34_empaquetar_operativo.py
```

The bundle contains M3, M2, propagation, calibrators, thresholds and D2
prevalence metadata. It excludes M1 and legacy artifacts.

### 5.4 Verify the frozen scientific candidate

```bash
python scripts/32_verificar_candidato.py
python scripts/33_verificar_trazabilidad_git.py
```

The check compares the 100 frozen artifacts against `auditoria/FASE5_CANDIDATO_CHECKSUMS.csv`.

## 6. Options and expected behaviour

- All thresholds used for evaluation are validation-derived; test thresholds are
  never recomputed.
- Missing thresholds, prevalence metadata or operational models raise an error
  instead of silently using defaults.
- The annual fuel join uses the prior-year map; 2024 uses 2023 (`lag 1`) and 2025
  reuses 2023 (`lag 2`).
- The spatial holdout is 50-km blocks with a 20-km buffer.
- The confirmation holdout is evaluated by `scripts/37_validacion_confirmacion_2023.py` after retraining without the confirmation blocks and buffer.

## 7. Troubleshooting

- `FileNotFoundError` for a model or threshold: run the corresponding training
  script or restore the v2.0.0 data archive.
- `ValueError` for D2 partitions: run `scripts/00_validar_datos.py` and inspect
  `tablas/tabla00_validacion_datos.csv`.
- CDS/FIRMS access errors: raw acquisition is outside the frozen evaluation
  package; use the archived test targets and probabilities for table-level
  reproduction.

## 8. Citation

Cite the code version DOI `10.5281/zenodo.23717175`, the data version DOI
`10.5281/zenodo.23717176` and the manuscript. Concept DOIs are also available in
`CITATION.cff`.
