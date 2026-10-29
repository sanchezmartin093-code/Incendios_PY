# Fase 0 - Acta de congelamiento

**Fecha:** 2026-10-27  
**Rama:** `audit/fase0-freeze`  
**Commit de congelamiento pre-auditoria:** `94d5b5e`  
**Tag pre-auditoria:** `pre-audit-annual-fuel-2026-10-27`  
**Tag de cierre:** `fase0-freeze-2026-10-27` (aplicado al commit que contiene esta acta)  
**Estado:** Fase 0 ejecutada; no se modificó código, modelos ni features.

## 1. Objetivo

Congelar el estado del proyecto después de la reconstrucción con fuel anual MapBiomas y antes de corregir las vulnerabilidades detectadas en la auditoría. El objetivo es que las correcciones posteriores sean auditables y que no se contamine más el conjunto de test.

## 2. Decisiones tomadas

1. **2024-2025 se declara out-of-time retrospectivo, no ciego.** El split de test actual incluye 2024 y 2025 y ya fue observado durante el desarrollo.
2. **No se usará 2024-2025 para seleccionar modelos, variables, umbrales ni hiperparámetros.**
3. **El holdout de confirmación será espacial**, porque 2026 no tiene cobertura suficiente al día de hoy: FIRMS llega hasta 2026-01-04 y el clima procesado hasta 2025-12-31.
4. **El holdout espacial nuevo debe ser independiente del holdout usado por `20_validacion_espacial_bloques.py`.**
5. **Los resultados actuales son provisionales.** No se enviará manuscrito ni se actualizará DOI hasta completar las correcciones y la confirmación.
6. **La comparación M3/M2 debe hacerse con la misma capacidad** o presentarse explícitamente como comparación de capacidad distinta.
7. **Los umbrales y fallbacks deben fallar cerrado:** si falta el metadato o el fuel anual, el pipeline no debe continuar silenciosamente con valores hardcodeados o estáticos.

## 3. Estado del repositorio

- El estado pre-auditoría quedó en el commit `94d5b5e`.
- Se incluyó `scripts/03a_procesar_combustible_anual.py`, que estaba sin trackear.
- Los datos procesados y los modelos están fuera de git por `.gitignore`; por eso se generó un manifiesto de hashes.
- No se hizo push ni se publicó una release nueva.

## 4. Artefactos de Fase 0

- `auditoria/FASE0_AMBIENTE.txt`: sistema, Python, versiones y `pip freeze`.
- `auditoria/FASE0_MANIFIESTO_HASHES.csv`: 188 archivos críticos con SHA-256.
- `auditoria/FASE0_METRICAS_BASE.md`: métricas base provisionales.
- `auditoria/FASE0_METRICAS_BASE.csv`: subconjunto tabular de métricas base.
- `auditoria/FASE0_ESTADO_TEST.md`: estado y reglas del test.
- `auditoria/FASE0_HOLDOUT_DECISION.md`: decisión y diseño del holdout nuevo.
- `auditoria/FASE0_PROTOCOLO_EVALUACION.md`: protocolo congelado.
- `auditoria/FASE0_REGISTRO_HALLAZGOS.csv`: hallazgos C1-C8 y H1-H10.
- `auditoria/FASE0_TESTS.txt`: salida de la suite (66 passed, 2 warnings) y vacios conocidos.
- `auditoria/FASE0_RELEASE_INVENTARIO.md`: brecha entre release actual y estado real.
- `auditoria/FASE0_ESTADO_COMUNICACION.md`: afirmaciones prohibidas hasta Fase 2.

## 5. Gates de Fase 0

- **Gate 0A - Congelamiento:** rama, commit, tag, ambiente y manifiesto de hashes creados.
- **Gate 0B - Test:** 2024-2025 declarado retrospectivo; holdout nuevo definido y bloqueado.
- **Gate 0C - Protocolo:** métricas, umbrales y criterios de selección definidos antes de mirar el holdout nuevo.
- **Gate 0D - Registro:** hallazgos críticos y altos documentados con evidencia e impacto.

## 6. Riesgos abiertos

- El manifiesto de datos `fuentes/MANIFIESTO_DATOS_v2.csv` está obsoleto y no debe usarse como fuente de verdad.
- El DOI actual `v1.0.0` no corresponde al estado anual.
- Los modelos y datos actuales no están en git; están protegidos por hashes y por el tag, pero no por una release nueva.
- El entorno de ejecución actual depende de `/tmp/auditvenv`; debe reconstruirse con un lock en Fase 3.
- El holdout espacial nuevo aún no está implementado; solo está definido y reservado.

## 7. Próxima fase

La Fase 1 no debe comenzar hasta que el protocolo y el holdout hayan sido aceptados. La Fase 1 corregirá, como mínimo:

1. el glob de MapBiomas en `03`;
2. la fuga de variables antecedentes en `21`;
3. los umbrales y fallbacks fail-open;
4. la inconsistencia de capacidad M2/M3;
5. el orden determinista de dataset y targets;
6. la falta de test de la lógica D2 de producción;
7. la regeneración de prevalencias.

**No se modificó código, modelo ni dato durante la Fase 0.**
