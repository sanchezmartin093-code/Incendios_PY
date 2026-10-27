# Fase 5 - Pendientes después de la reauditoría

**Fecha:** 2026-10-27  
**Candidato vigente:** `submission-candidate-2026-10-27-r2` → `7390cca`  
**HEAD de documentación:** `f4c2350`  
**Estado:** código verificado; Fase 5 no ejecutada.

## 1. Bloqueantes antes de reentrenar

1. **Corregir trazabilidad Git/tag del candidato.**  
   `FASE5_CANDIDATO_CHECKSUMS.json` registra el archive de `e621f95`, pero el tag r2 apunta a `7390cca`. Hay que separar `base_commit`, `tag_commit`, `tag_tree` y `tag_archive_sha256`.

2. **Completar validación de artefactos críticos en `scripts/31`.**  
   Hoy exige solo M3/M2 Chaco. Debe exigir M3/M2 Oriental, propagación Chaco/Oriental, calibradores y umbrales de ambas regiones.

3. **Agregar verificación Git/tag a `scripts/32`.**  
   Debe verificar tag, commit, tree y archive, no solo archivos y corpus.

4. **Quitar `host` del JSON público del candidato.**  
   Evitar publicar el hostname de la máquina.

## 2. Bloqueantes científicos

5. **Ejecutar la Fase 5 una sola vez.**  
   Reentrenar excluyendo holdout + buffer, recalcular priors, calibrar y fijar cutoffs solo con desarrollo 2023, y evaluar el holdout en 2024–2025.

6. **Reportar confirmación.**  
   Crear `FASE5_CONFIRMACION.md`, `FASE5_CONFIRMACION.json` y `tabla13_confirmacion_*`.

7. **Actualizar el manuscrito con confirmación.**  
   Si el resultado es negativo, suavizar o cambiar las afirmaciones; no reajustar el modelo.

## 3. Bloqueantes de release

8. **Generar `MANIFIESTO_DATOS_v3.csv`.**  
   El v2 contiene dataset monolítico viejo, perfil sintético y targets deprecados.

9. **Actualizar metadatos.**  
   `CITATION.cff`, `.zenodo.json`, `title_page.qmd`, `cover_letter.qmd`, README y manuscrito siguen con DOI/versión/fecha pre-auditoría.

10. **Resolver o aceptar formalmente el lock sin hashes.**  
    `requirements-lock.txt` tiene versiones exactas y 0 hashes de ruedas.

11. **Generar PDFs finales.**  
    Falta `manuscrito.pdf`; `material_suplementario.pdf` es anterior a la reconciliación.

12. **Merge/push y release.**  
    `main` está detrás de la rama; no existe `v2.0.0`; no hay DOI nuevo; no se hizo push.

## 4. Mejoras no bloqueantes

13. Excluir buffer también en la calibración si se reutiliza `scripts/27`.
14. Agregar aserciones para el rol `buffer_confirmacion` en tests.
15. Extender CI a manuscrito ES/EN y suplementos.
16. Evaluar si `00_validar_datos.py` debe validar el contenido del manifiesto v3, no solo que existan hashes.

## 5. Regla

No se reentrena ni se toca el holdout hasta cerrar los puntos 1–4 y congelar un candidato r3 si hay cambios de código.
