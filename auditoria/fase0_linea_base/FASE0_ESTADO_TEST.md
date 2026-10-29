# Fase 0 - Estado y congelamiento del test

**Fecha:** 2026-10-27  
**Commit:** `94d5b5e`  
**Tag:** `pre-audit-annual-fuel-2026-10-27`

## 1. Split actual

El split de producción está en `scripts/05d_muestrear_poblacional_d2.py`:

- **Train:** 2018-2022.
- **Validation:** 2023.
- **Test actual:** 2024-2025.

El test contiene 376.552 celdas-día de ignición y 1.437.414 de propagación; 1.813.966 celdas-día-tarea en total.

## 2. Problema

El test 2024-2025 **ya fue observado** durante el desarrollo:

- se usó para evaluar el fuel estático;
- se usó para evaluar el fuel anual;
- se usó para comparar modelos, features y ablaciones;
- se usó para decidir qué resultados mostrar.

Por lo tanto:

- **no es un test ciego**;
- **no es un holdout prístino**;
- no puede sostener la afirmación “no test row is used for model selection”;
- no puede usarse para elegir C3, ET0, variables, arquitectura o umbrales.

## 3. Decisión

1. 2024-2025 queda declarado **out-of-time retrospectivo**.
2. Se prohíbe volver a evaluar sobre 2024-2025 hasta que exista un protocolo congelado.
3. El holdout de confirmación será un **nuevo holdout espacial** (ver `FASE0_HOLDOUT_DECISION.md`).
4. 2026 queda como validación prospectiva futura, solo si se completa la cobertura de labels, clima y fuel.
5. El test retrospectivo puede usarse para diagnóstico y para comparar con la literatura, pero no como confirmación final.

## 4. Regla de uso

| Uso | 2024-2025 retrospectivo | Holdout nuevo |
|---|---:|---:|
| Diagnóstico interno | Permitido | Prohibido hasta Fase 5 |
| Comparación con literatura | Permitido con etiqueta | Prohibido hasta Fase 5 |
| Selección de variables | Prohibido | Prohibido |
| Selección de umbrales | Prohibido | Prohibido |
| Selección de modelo | Prohibido | Prohibido |
| Confirmación final | No | Sí, una sola vez |

## 5. Consecuencia para el manuscrito

Toda mención a “blind test 2024-2025” debe cambiarse por “retrospective out-of-time test 2024-2025” o equivalente. Si se mantiene la palabra “blind”, el revisor puede argumentar correctamente que el test fue observado.

## 6. Criterio de cierre

Este memo se considera cumplido cuando:

- el manuscrito deja de llamar “ciego” al test 2024-2025;
- el holdout nuevo está definido y bloqueado;
- ninguna decisión de Fase 1 usa 2024-2025;
- la confirmación final se hace una sola vez sobre el holdout nuevo.
