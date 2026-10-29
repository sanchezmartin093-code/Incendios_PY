# Fase 0 - Estado de comunicacion

**Fecha:** 2026-10-27  
**Estado:** resultados provisionales; comunicacion congelada.

## 1. Mensaje interno

El pipeline fue reconstruido con fuel anual MapBiomas y los modelos fueron reentrenados. Los resultados actuales son **provisionales** y no deben presentarse como finales. La prioridad es corregir la fuga de lead time, el glob de MapBiomas, los umbrales fail-open y la inconsistencia de capacidad antes de comunicar resultados.

## 2. Afirmaciones prohibidas hasta Fase 2

- “El test 2024-2025 es ciego.”
- “El modelo no usa informacion futura” sin aclarar que 2025 usa fuel 2023.
- “M3 pierde transferibilidad espacial” sin reconciliar la tabla 8 actual.
- “El codigo y los datos actuales estan archivados con DOI.”
- “La comparacion M3 vs M2 aisla el efecto de las features” con capacidades distintas.
- “El lead time es de 24 h” sin corregir las features antecedentes.
- “El modelo esta listo para operacion” mientras existan fallbacks silenciosos.

## 3. Afirmaciones permitidas

- “Se incorporo fuel anual MapBiomas con asignacion Y-1.”
- “El pipeline separa ignicion y propagacion.”
- “Existe un test retrospectivo 2024-2025 y un protocolo en revision.”
- “Los resultados actuales son provisionales y estan siendo auditados.”
- “El sistema es un nowcast, no un pronostico NWP.”

## 4. Audiencia

- **Interna:** puede ver metricas provisionales con la etiqueta correspondiente.
- **Revisores/periodistas:** solo despues de Fase 2 y con el holdout nuevo.
- **Instituciones operativas:** solo despues de Fase 1 y con fallback fail-closed.

## 5. Criterio de cierre

- Manuscrito sin lenguaje de “blind test” para 2024-2025.
- README y supplement sin metricas viejas.
- DOI actualizado solo despues de release.
- Aprobacion explicita antes de cualquier envio.
