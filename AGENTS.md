# AGENTS.md — Reproducible Research Project

This repository follows the `Proyectos_PY` scientific rigor standard (Q1 / Nature 2026).
The full, extended specification lives in the workspace at `docs/estandares/` when
available; this file is the self-contained contract for this project.

## 1. Non-negotiable rules

1. **Clean root.** Only configuration, administration and canonical first-level
   directories. Never mix sources with generated artifacts.
2. **Raw data is immutable.** `datos/raw/` is never edited, overwritten or deleted.
3. **One-command reproduction.** `make all` rebuilds every deliverable. Never edit
   generated files by hand.
4. **Zero-Computation in manuscript.** `.qmd` files must not fit models, run MCMC or
   perform heavy transforms. Compute in `scripts/`, then inject results.
5. **Zero Hardcoded Numbers.** Never type a statistic manually in narrative text.
   Inject every estimate from `datos/outputs/metricas_clave.json`.
6. **Native Unicode, minimal LaTeX.** Use `R²`, `β`, `χ²`, `α`, `±`, `95% CI`,
   `p < 0.001`. Inline LaTeX and `\textbf{}`/`\cite{}` are forbidden. Reserve
   `$$...$$` for complex multiline formulas.
7. **Root-anchored paths.** `_quarto.yml` uses `execute-dir: project`. Use `/` prefix
   in Markdown/YAML resources; use project-relative paths inside R/Python code.
8. **Deterministic computation.** Keep `TZ=UTC`, `LC_ALL=C.UTF-8` and
   `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`. In R: `RNGkind("L'Ecuyer-CMRG")` and
   `set.seed(seed_master)`. In Python: `numpy.random.default_rng(seed)`.
9. **Data contracts.** Validate schemas, types and ranges in `01_preprocesamiento`
   (`pointblank` in R, `pandera` in Python). Abort on schema drift.
10. **Privacy.** Human data must be de-identified (HIPAA Safe Harbor) before sharing.
    Identifiable raw data stays in `_privado/` (always gitignored).
11. **Git size gate.** Never track files larger than 50 MB. Large assets live in
    Zenodo/GitHub Releases with a verified SHA-256.
12. **Interoperable models.** No `.pkl`/`.joblib`. Use ONNX or parameter tables for
    ML; export Bayesian MCMC draws to Parquet (`posterior::as_draws_df`).
13. **Vector figures first.** Primary figures are `.pdf`/`.svg` in
    `entregables/fuentes/figuras/`; `.png` copies live in `figuras/png/`.
14. **Strict CSV.** RFC 4180: comma delimiter, dot decimal, UTF-8 without BOM.
15. **Reporting guideline.** Complete `docs/CHECKLIST_<GUIDE>.md` with real page and
    line numbers from the final DOCX. Freeze `docs/SAP.md` before analysing outcomes.

## 2. Canonical commands

```bash
make setup       # Install the locked environment (pixi or renv)
make data        # Pull/verify raw data and hashes
make pipeline    # Run the targets DAG; refresh metrics and hashes
make test        # Run R and Python unit tests
make bib-check   # Validate references.bib integrity and DOIs
make audit       # Manuscript, editorial and infrastructure gates
make render      # Compile identified DOCX/HTML (Zero-Computation)
make anonymous   # Compile the blinded manuscript into revision_anonima/
make mcp-test    # Self-test the Paper2Agent MCP server
make all         # End-to-end reproduction
```

## 3. Canonical layout

```text
config.yml            # Seed, design, EQUATOR guide, editorial profile
data_contract.yml     # Raw-data schema, types and plausible ranges
_targets.R / R/       # Analytical DAG and pipeline functions
scripts/              # I/O, gates, export, editorial post-processing, MCP
manuscrito/           # Single-source .qmd (en/ required; es/ optional for dissemination)
entregables/          # Compiled deliverables and vector figures/tables
datos/                # raw/ (immutable), processed/, outputs/metricas_clave.json
docs/                 # SAP.md, CHECKLIST_<GUIDE>.md, protocols
auditoria/            # HASHES.md + HASHES.sha256 and phase evidence folders
modelos/              # ONNX / parameter tables / MCMC draws (never .pkl)
tests/                # Unit tests and synthetic fixtures (datos_mock/)
```

## 4. Agent behaviour

- Respond to the user in **Spanish**; write code, comments, docstrings, commits and
  technical documentation in **English**.
- Read `docs/SAP.md`, `docs/CHECKLIST_<GUIDE>.md` and `config.yml` before changing
  analyses or narrative.
- Prefer editing sources and re-running `make` over patching generated artifacts.
- Run `make test`, `make audit` and the relevant linters after changes. Never commit
  unless the user explicitly asks.
