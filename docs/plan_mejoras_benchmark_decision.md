# Plan de mejoras: blindaje del benchmark (B) y valor de decisión (C)

**Proyecto:** `Incendios_PY` — manuscrito para *Computers & Geosciences*
**Fecha:** 2026-10-28
**Alcance:** mejoras B (comparación con FWI/INPE) y C (sustituir la narrativa de AUC por beneficio de decisión).
**Fuera de alcance:** palanca D (test ciego 2026), aplazada por su coste temporal; validación externa, tratada en documento aparte.

> Nota de rutas: el repositorio fue reorganizado. Las fuentes del manuscrito están en `manuscrito/en/` y `manuscrito/es/`; las tablas generadas se publican en `entregables/tablas/`. Los scripts conservan su numeración original en `scripts/`.

---

## 1. Resumen ejecutivo

| Palanca | Qué ataca | Coste | Ganancia estimada |
|---|---|---|---|
| **B. Blindar FWI/INPE** | Crítica de comparación injusta ("straw man") | 1–2 semanas | +5 pp |
| **C. Valor de decisión** | Objeción "¿para qué sirve con precisión 0.08 %?" | 2–3 semanas | +3–5 pp |

Las ganancias en puntos porcentuales son incrementos subjetivos de probabilidad de aceptación y **no son aditivos** entre sí. B y C se refuerzan mutuamente: los mismos artefactos (FWI/INPE dentro del análisis de decisión) sirven a ambas.

---

## 2. Estado actual verificado en el código

### 2.1 Comparación con FWI/INPE

| Punto | Estado real | Evidencia |
|---|---|---|
| FWI completo (FFMC → DMC → DC → BUI → ISI → FWI) | **Ya implementado** | `src/fwi.py`, `scripts/02d_calcular_fwi_completo.py` |
| INPE Risco de Fogo | **Ya implementado** | `src/inpe.py`, `scripts/02e_calcular_inpe.py` |
| Umbral de FWI e INPE elegido en validación con F-beta 0.5 | **Ya implementado** | `scripts/13_benchmarking_modelos.py` (`calcular_umbral_optimo_f_beta`) |
| Mismas filas de test para todos los modelos e índices | **Ya implementado** | join por `nodo_clima_id` + `fecha` |
| Declarar la simetría del umbral en el manuscrito | **No hecho** | — |
| Baseline "solo meteorología" | **No existe** | — |
| Baseline "índice + contexto local" | **No existe** | — |
| FWI/INPE dentro del análisis cost-loss | **No existe** | — |
| Documentación metodológica de los índices en el suplemento | **No existe** | — |

### 2.2 Valor de decisión

| Punto | Estado real | Evidencia |
|---|---|---|
| Cost-loss normalizado V(α) con corrección de prevalencia | **Ya implementado** | `src/cost_loss.py`, `scripts/22_analisis_cost_loss.py` |
| Pesos poblacionales por fecha y región | **Ya implementado** | `_pesos()` en `scripts/22` |
| Curvas e informe (Tabla S14, Figura S3) | **Ya existe** | `tablas/tabla11_cost_loss.csv` |
| Intervalos de confianza sobre V(α) | **No existe** | — |
| Beneficio neto (decision curve analysis) | **No existe** | — |
| Métricas de carga operativa | **No existe** | — |
| FWI/INPE dentro del cost-loss | **No existe** | — |
| Test estadístico de diferencia de valor M3 vs M2 | **No existe** | — |

---

## 3. Palanca B — Blindar la comparación con FWI/INPE

Objetivo: que un revisor no pueda atribuir la ventaja de M3 a una comparación desleal (resolución, umbral o información disponible).

### B.1 Publicar la simetría de umbral

**Estado:** completado (2026-10-28). La tabla principal, la Tabla S11 y los textos EN/ES reportan ahora `Cutoff`, `Precision`, `Recall` y `F0.5` para FWI e INPE, con la regla F0.5 declarada.

**Qué hacer**
1. Exportar `cutoff_f05` de FWI e INPE en el CSV del benchmark.
2. Añadir a la tabla principal las columnas `umbral_f05`, `precisión`, `recall`, `F0.5` de cada índice.
3. Declarar en el suplemento: *"FWI and INPE cutoffs were selected on 2023 validation with the same F0.5 rule applied to the models and then frozen for 2024–2025."*

**Archivos:** `scripts/13_benchmarking_modelos.py`, `entregables/tablas/tabla04_benchmarking_internacional.csv`.

**Salida:** tabla de benchmark ampliada.

**Criterio de aceptación:** la tabla permite verificar que ningún índice se evaluó con un umbral más favorable que los modelos.

**Riesgo:** ninguno; es documental.

---

### B.2 Baseline "solo meteorología"

**Qué hacer**
- Entrenar un GBT con exactamente las variables que alimentan al FWI: temperatura, humedad relativa, viento, precipitación (y opcionalmente FFMC/ISI), sin combustible, sin accesibilidad, sin priors y sin calendario.
- Mismo esquema de entrenamiento, misma validación 2023 y mismo cutoff F0.5.

**Archivos**
- Nuevo: `scripts/43_benchmark_weather_only.py`.
- Reutiliza `src/modelo.py` con una lista `FEATURES_WEATHER_ONLY`.

**Salida:** fila adicional en `tabla04` y en la figura del benchmark.

**Interpretación**

| Resultado | Lectura |
|---|---|
| Weather-only ≈ FWI (0.51–0.56) | El problema es la **información**, no el algoritmo |
| Weather-only ≫ FWI | El problema es la **parametrización** del índice canadiense |

Ambos resultados son publicables.

**Criterio de aceptación:** existe una fila "weather-only GBT" en el benchmark y el texto la interpreta.

---

### B.3 Referencia intermedia "índice + contexto local"

**Qué hacer**
- Entrenar un GBT con FWI e INPE como entradas, más las features locales de M3 (combustible, accesibilidad, densidad histórica).

**Salida:** fila adicional en el benchmark.

**Interpretación**
- Si ≈ M3: el índice es útil pero insuficiente por sí solo; reformular el mensaje central.
- Si ≪ M3: la información local de alta resolución es la que domina.

**Riesgo:** si empata con M3, hay que revisar el título y el abstract para no vender "M3 supera a FWI" como hallazgo principal.

---

### B.4 Resolver la ambigüedad del INPE implementado

**Problema detectado:** la constante A del INPE se calcula con `constante_a_desde_fracciones(...)` a partir de `pct_pastura`, `pct_sabana_pastizal`, `pct_bosque_seco`, `pct_bosque_humedo` y `pct_humedal_palmar` (`scripts/13_benchmarking_modelos.py`). Es decir, el INPE implementado **ya incorpora información local de cobertura**.

**Qué hacer (elegir una)**
1. **Recomendada:** mantener A local como versión primaria y añadir una versión con **A fija** como sensibilidad.
2. Usar A fija como primaria y la local como sensibilidad.

**Corrección de texto obligatoria en 4.1:** hoy afirma que los índices son "purely meteorological". Si el INPE usa fracciones de cobertura, la frase es inexacta. Reformular:

> "FWI is purely meteorological; the INPE formulation used here incorporates land-cover fractions, yet neither carries the fine-scale anthropogenic and accessibility information that dominates ignition in the Chaco."

**Criterio de aceptación:** el texto y el suplemento describen exactamente lo implementado.

---

### B.5 Nota S4 — Implementación de los índices

Añadir una nota suplementaria con:
1. Ecuaciones y códigos de combustible del FWI.
2. Origen de las variables de mediodía (ERA5-Land 3-h).
3. Uso de meteorología del **mismo día** (nowcast), no t-1.
4. Inicialización (85/6/15) y warm-up.
5. Versión del INPE (2019), definición de PSE (ventana 120 d) y de la constante A.
6. Declaración explícita: *"no local calibration was applied to either index"*.
7. Resolución: 0.25° por nodo ERA5, asignada a celdas de 1 km por `nodo_clima_id`.

**Salida:** `manuscrito/en/material_suplementario.qmd` (+ versión ES).

**Criterio de aceptación:** un revisor puede replicar ambos índices solo con la nota y el código.

---

### B.6 Opcional — FWI a 1 km

Interpolar las variables meteorológicas a 1 km y recalcular el FWI por celda. Si el FWI a 1 km sigue en ~0.52, la crítica de escala queda descartada definitivamente.

**Coste:** 3–5 días.

---

## 4. Palanca C — Valor de decisión

Objetivo: que la utilidad operativa sea un resultado primario y no una comprobación secundaria.

### C.1 Intervalos de confianza bootstrap sobre V(α)

**Qué hacer**
- Para cada α y cada modelo, remuestrear **fechas** (no filas), B = 1 000, recalcular V(α) y reportar IC 95 %.
- Reutilizar el patrón de `_delta_bootstrap` de `scripts/26_validacion_2023.py`.

**Salida:** `Tabla S14` con `valor_lo` y `valor_hi`; `Figura S3` con bandas.

**Criterio de aceptación:** ninguna afirmación de superioridad sin IC.

---

### C.2 Curvas de beneficio neto (decision curve analysis)

**Definición**

```
NB(p_t) = TP_w/N_w − (FP_w/N_w) · p_t/(1 − p_t)
```

- Eje x: probabilidad umbral `p_t` (0–0.30 ignición; 0–0.90 propagación).
- Eje y: beneficio neto en detecciones por 1 000 celdas-día.
- Referencias obligatorias: no alertar (NB = 0), alertar todo, climatología.
- Curvas por modelo: M1, M2, M3, M3/M2 gated, propagación, **FWI**, **INPE**.

**Salida:** `Figura S4` (nueva) y columnas en la tabla de decisión.

**Criterio de aceptación:** se puede comparar M3 con FWI/INPE en el mismo eje de decisión.

---

### C.3 Métricas de carga operativa

| Métrica | Definición |
|---|---|
| `alertas_dia` | Celdas alertadas por día a la prevalencia real |
| `recall_top_k` | Recall con presupuesto de K alertas/día (K = 10, 50, 100) |
| `precision_top_k` | Precisión en ese presupuesto |
| `alertas_por_acierto` | Alertas necesarias para detectar un incendio (1/precisión) |
| `lift_climatologia` | Precisión / prevalencia |
| `alpha_optimo` | α que maximiza V(α) por modelo y región |

**Salida:** `Tabla S22` (nueva).

**Criterio de aceptación:** el manuscrito traduce la precisión poblacional a una carga operativa concreta.

---

### C.4 Incluir FWI e INPE en el análisis de decisión

**Qué hacer**
- Pasar `p_fwi` y `p_inpe` (ya calculadas en `scripts/13`) a `valor_cost_loss` y a beneficio neto, con sus cutoffs congelados.

**Salida:** filas de FWI/INPE en `Tabla S14`, `Tabla S22` y `Figura S4`.

**Criterio de aceptación:** se responde si la ventaja en AUC se traduce en valor de decisión.

---

### C.5 Test pareado de diferencia de valor

**Qué hacer**
- Bootstrap pareado por fecha de `V_M3(α) − V_M2(α)` y `V_M3(α) − V_FWI(α)`.
- Reportar IC y el rango de α donde el IC excluye 0.

**Salida:** `Tabla S23` (nueva) o columnas en `Tabla S22`.

**Criterio de aceptación:** la afirmación "M3 no supera claramente a M2" pasa de cualitativa a cuantificada.

---

### C.6 Reordenar la narrativa

1. Renombrar §3.4 como **"Operational value"**; cost-loss y beneficio neto como resultados primarios.
2. Mover calibración (conformal, ECE) a **§3.5 "Calibration"**.
3. En **4.1**, cerrar con: *"the indices are not only weaker rankers; they also generate less decision value at any plausible cost-loss ratio."*
4. En **4.2**, incluir el eje de decisión en la tríada ranking/bloques/prevalencia.
5. En **Conclusiones**, una frase con el resultado de decisión (propagación con valor positivo para α ≤ 0.005; ignición solo como cribado).
6. **Abstract:** está en 245/250 palabras; no añadir, **sustituir**. Propuesta:

> "Population-level ignition precision is 0.082 %/0.198 % and cost-loss value is positive only for propagation, confirming a screening-and-prioritization role rather than a stand-alone warning system."

---

## 5. Sinergia B ∩ C

| Artefacto | Contenido | Coste | Impacto |
|---|---|---|---|
| **Tabla benchmark ampliada** | M1/M2/M3/gated/prop/weather-only/FWI/INPE/A-fija, con cutoff, P/R/F0.5 | 3 días | Alto |
| **Tabla de decisión** | V(α) con IC, NB, alertas/día, recall@k, α óptimo, incluidos FWI/INPE | 5 días | Muy alto |
| **Figura S4** | Beneficio neto por región y tarea, con bandas | 2 días | Alto |
| **Nota S4** | Implementación de FWI e INPE y declaración de no calibración local | 1 día | Medio-alto |

---

## 6. Cronograma

| Semana | Tarea | Entregable |
|---|---|---|
| 1 | B.1 + B.5 | Tabla de benchmark con umbrales; Nota S4 |
| 1–2 | B.2 | Fila weather-only; decisión sobre el mensaje |
| 2 | B.4 | Sensibilidad A fija vs A local del INPE |
| 2–3 | C.1 + C.2 + C.4 | Tabla S22 + Figura S4 |
| 3 | C.3 + C.5 | Tabla S23 |
| 3–4 | Reescritura §3.4/§3.5, 4.1, 4.2, Conclusiones, Abstract | Manuscrito v3 |
| 4 (opcional) | B.3 + B.6 | Tabla S24 / Figura S5 |

**Total:** 3–4 semanas de trabajo efectivo, sin cómputo pesado (cabe en el portátil con DuckDB a 9 GB).

---

## 7. Checklist de terminado

- [x] FWI e INPE tienen `cutoff_f05`, precisión, recall y F0.5 en la tabla principal.
- [ ] Existe un baseline weather-only GBT y está en la tabla.
- [ ] Existe una versión canónica (A fija) del INPE, o el texto declara que A es local.
- [ ] El texto de 4.1 ya no afirma que ambos índices son "purely meteorological" sin matiz.
- [ ] La Nota S4 documenta ecuaciones, inicialización, resolución y ausencia de calibración local.
- [ ] V(α) tiene IC bootstrap en tabla y figura.
- [ ] Beneficio neto con referencias (no alertar / alertar todo / climatología) para ignición y propagación.
- [ ] FWI e INPE aparecen en el análisis de decisión.
- [ ] Métricas de carga operativa (alertas/día, recall@k, alertas por acierto, lift).
- [ ] Test pareado de ΔV(M3−M2) y ΔV(M3−FWI).
- [ ] Abstract ≤ 250 palabras y refleja el mensaje de decisión.
- [ ] Conclusiones mencionan el valor operativo, no solo el AUC.
- [ ] Versión ES sincronizada.

---

## 8. Resultado esperado

Con B y C cerrados, el manuscrito deja de apoyarse en el ΔAUC de +0.003 entre M3 y M2 y pasa a defender:

> el índice canónico falla, el protocolo D2 lo demuestra con una comparación simétrica, y el valor operativo está en la propagación y en el cribado de ignición.

Esa tesis es más resistente a revisión que una afirmación de superioridad marginal del modelo híbrido.
