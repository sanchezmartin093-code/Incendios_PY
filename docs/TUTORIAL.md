# Quick tutorial: reproduce one table and one alert

This tutorial assumes the v2.1.0 data archive is placed next to the repository.

## A. Reproduce the gated M3/M2 table

```bash
python scripts/35_evaluar_politica_gated.py
cat entregables/tablas/tabla13_politica_gated.md
```

Expected outputs:

- `entregables/tablas/tabla13_politica_gated.csv`
- `entregables/tablas/tabla13_politica_gated_poblacional.csv`
- `entregables/tablas/tabla13_politica_gated.md`

The script selects the gated F0.5 cutoff on 2023 validation and applies it to
the retrospective 2024–2025 test.

## B. Reproduce the 2023 validation table

```bash
python scripts/26_validacion_2023.py
```

Expected outputs in `entregables/tablas/tabla10_validacion_2023.csv` and related files.

## C. Run one daily alert

```bash
python scripts/09_servicio_alerta.py --fecha 2024-09-09
```

Inspect `datos/outputs/boletin_alerta_2024-09-09.md`.

## D. Verify the frozen candidate

```bash
python scripts/32_verificar_candidato.py
```

Expected final line: `OK: checksums, tamaños y corpus coinciden.`
