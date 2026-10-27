# Fase 5 - Pendientes después de la reauditoría

**Fecha:** 2026-10-27
**Candidato vigente:** `submission-candidate-2026-10-27-r2` → `7390cca`
**Estado:** trazabilidad Git/tag corregida; código verificado; Fase 5 no ejecutada.

## 1. Bloqueantes antes de reentrenar

1. **Completar validación de artefactos críticos en `scripts/31`.**
   Hoy exige solo M3/M2 Chaco. Debe exigir M3/M2 Oriental, propagación Chaco/Oriental, calibradores y umbrales de ambas regiones.

2. **Integrar la verificación Git/tag en `scripts/32`.**
   Ya existe `scripts/33_verificar_trazabilidad_git.py` como verificador externo;
   conviene incorporar esa lógica a `scripts/32` en el próximo candidato.

## 2. Resuelto en esta corrección

- Se separaron `source_commit`, `source_tree`, `source_archive_sha256`,
  `tag_commit`, `tag_tree` y `tag_archive_sha256`.
- El tag r2 conserva el commit `7390cca` y su archive `a768...`.
- El snapshot científico sigue siendo `e621f95`, archive `6988...`.
- Se quitó `host` del JSON público.
- Se agregó el verificador de trazabilidad `scripts/33_verificar_trazabilidad_git.py`.

## 3. Bloqueantes científicos

3. **Ejecutar la Fase 5 una sola vez.**
   Reentrenar excluyendo holdout + buffer, recalcular priors, calibrar y fijar cutoffs solo con desarrollo 2023, y evaluar el holdout en 2024–2025.

4. **Reportar confirmación.**
   Crear `FASE5_CONFIRMACION.md`, `FASE5_CONFIRMACION.json` y `tabla13_confirmacion_*`.

5. **Actualizar el manuscrito con confirmación.**
   Si el resultado es negativo, suavizar o cambiar las afirmaciones; no reajustar el modelo.

## 4. Bloqueantes de release

6. **Generar `MANIFIESTO_DATOS_v3.csv`.**
   El v2 contiene dataset monolítico viejo, perfil sintético y targets deprecados.

7. **Actualizar metadatos.**
   `CITATION.cff`, `.zenodo.json`, `title_page.qmd`, `cover_letter.qmd`, README y manuscrito siguen con DOI/versión/fecha pre-auditoría.

8. **Resolver o aceptar formalmente el lock sin hashes.**
   `requirements-lock.txt` tiene versiones exactas y 0 hashes de ruedas.

9. **Generar PDFs finales.**
   Falta `manuscrito.pdf`; `material_suplementario.pdf` es anterior a la reconciliación.

10. **Merge/push y release.**
    `main` está detrás de la rama; no existe `v2.0.0`; no hay DOI nuevo; no se hizo push.

## 5. Mejoras no bloqueantes

11. Excluir buffer también en la calibración si se reutiliza `scripts/27`.
12. Agregar aserciones para el rol `buffer_confirmacion` en tests.
13. Extender CI a manuscrito ES/EN y suplementos.
14. Evaluar si `00_validar_datos.py` debe validar el contenido del manifiesto v3, no solo que existan hashes.

## 6. Regla

No se reentrena ni se toca el holdout hasta cerrar los puntos 1–2 y congelar un candidato r3 si hay cambios de código.
