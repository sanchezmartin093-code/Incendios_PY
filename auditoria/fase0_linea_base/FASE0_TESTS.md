# FASE 0 - REGISTRO DE TESTS

fecha_utc: 2026-10-27T20:08:02.171425+00:00
commit: 591cfc80f6a113d42f5ae5081a00ab5af0ca9c63
comando: /tmp/auditvenv/bin/python -m pytest -q tests
returncode: 0

## stdout
..................................................................       [100%]
=============================== warnings summary ===============================
tests/test_targets.py::TestTargetsD2::test_definicion_d2
tests/test_targets.py::TestTargetsD2::test_dias_desde_fuego
  /Users/martinsanchez/Documents/Proyectos_PY/Incendios_PY/src/targets.py:46: UserWarning: Sortedness of columns cannot be checked when 'by' groups provided
    .join_asof(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
66 passed, 2 warnings in 3.01s

## stderr


## Vacios conocidos
- No hay test de integracion de la logica D2 de produccion en 05d frente a src/targets.py.
- No hay test de no-fuga de fuel anual en 04/05d/05e.
- No hay test de no-fuga de las features antecedentes en 21.
- No hay test de fail-closed de umbrales/fallback en 07/alerta.
- No hay test de reproducibilidad de orden en 04/05d/06/16.
