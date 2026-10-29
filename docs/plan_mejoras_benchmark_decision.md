# Plan de mejoras: blindaje del benchmark (B) y valor de decisión (C)

**Proyecto:** `Incendios_PY` — manuscrito para *Computers & Geosciences*

**Fecha:** 2026-10-28

**Alcance:** mejoras B (comparación con FWI/INPE) y C (sustituir la narrativa de AUC por beneficio de decisión).

**Fuera de alcance:** palanca D (test ciego 2026), aplazada por su coste temporal; validación externa, tratada en documento aparte.

> **Actualización 2026-10-28:** B.1–B.5, C.1–C.6 y R.1–R.12 están ejecutados. R.8 cubre confianza alta y FRP; la variante de nubes < 40 % no completó porque la cola del CDS no entregó el proxy ERA5. R.9 ya incluye una muestra sin gate y R.4 un 1:20 con negativos independientes. R.12 se verificó en entorno limpio (76 tests, hashes OK), E.1 (Boquerón) está completada y G0-Formosa se ejecutó como validación externa condicional (transporte limitado; ROC 0.71–0.72). Solo queda **B.6**, opcional.

> Nota de rutas: el repositorio fue reorganizado. Las fuentes del manuscrito están en `manuscrito/en/` y `manuscrito/es/`; las tablas generadas se publican en `entregables/tablas/`. Los scripts conservan su numeración original en `scripts/`.

---

## 1. Resumen ejecutivo

| Palanca | Qué ataca | Coste | Ganancia estimada |
|---|---|---|---|
| **B. Blindar FWI/INPE** | Crítica de comparación injusta ("straw man") | 1–2 semanas | +5 pp |
| **C. Valor de decisión** | Objeción "¿para qué sirve con precisión 0.08 %?" | 2–3 semanas | +3–5 pp |
| **R. Robustez** | Estabilidad, fuga residual, ruido de etiquetas, multiplicidad y reproducibilidad | 3–4 semanas | +4–6 pp |

Las ganancias en puntos porcentuales son incrementos subjetivos de probabilidad de aceptación y **no son aditivos** entre sí. B y C se refuerzan mutuamente: los mismos artefactos (FWI/INPE dentro del análisis de decisión) sirven a ambas. R convierte cada objeción previsible en una respuesta cuantitativa ya presente en el manuscrito y protege las ganancias de B y C.

---

## 2. Ruta priorizada (orden de ejecución)

Criterio de orden: primero lo que **no requiere entrenamiento** y puede cambiar el manuscrito en horas; después **un único lote de entrenamiento** (para amortizar el entorno y no reconfigurarlo varias veces); dentro del lote, primero el experimento decisivo (**R.2**), porque si el backtest contradice el mensaje hay que reescribir. Las tareas opcionales quedan al final.

### 2.1 Ya realizado

| # | Tarea | ¿Entrena? | Estado |
|---|---|---|---|
| 0 | B.1 Simetría de umbral FWI/INPE | No | Hecho |
| 0 | B.2 Baseline weather-only | Sí | Hecho |

Además están cerrados: **B.3** (índice + contexto local), **B.4** (A fija vs local), **B.5** (Nota S4), **C.1–C.6** (IC, DCA, carga operativa, índices en decisión, ΔV, narrativa), **R.1** (semillas/capacidad), **R.2** (rolling-origin), **R.3** (solapamiento), **R.5** (umbral/métrica), **R.6** (multiplicidad), **R.7** (adversarial), **R.11** (IC poblacionales) y **R.12** (hashes/`run_all.sh`).

> Las tablas §2.2–§2.5 conservan el orden original como registro. El estado real es el de §2.1 y el orden de lo pendiente es el de §2.7.

### 2.2 Fase 1 — Decisiones rápidas sin entrenamiento (3 días)

| #   | Tarea                                        | ¿Entrena? | Días | Por qué ahora                                                          |
| --- | -------------------------------------------- | --------- | ---- | ---------------------------------------------------------------------- |
| 1   | **R.6** Holm–Bonferroni / BH                 | No        | 1    | Máximo retorno por hora; elimina la objeción de p-hacking              |
| 2   | **R.3** Métricas sin solapamiento train/test | No        | 1    | Es la limitación más citada; se resuelve con predicciones ya guardadas |
| 3   | **R.11** IC de precisión poblacional         | No        | 1    | Convierte 0.082 %/0.198 % en una cifra defendible                      |

### 2.3 Fase 2 — Valor de decisión y texto (5 días)

| # | Tarea | ¿Entrena? | Días | Por qué ahora |
|---|---|---|---|---|
| 4 | **C.1 + C.5** IC de V(α) y ΔV(M3−M2/FWI) | No | 2 | Usa las predicciones existentes |
| 5 | **C.2 + C.4** Beneficio neto con FWI/INPE | No | 2 | Convierte el benchmark en valor operativo |
| 6 | **C.3** Carga operativa (alertas/día, recall@k) | No | 1 | Traduce la precisión a decisiones |
| 7 | **B.4** INPE con A fija vs A local | No (recalcula índice) | 1 | Cierra la ambigüedad metodológica |
| 8 | **B.5** Nota S4 de implementación | No | 1 | Documenta lo que ya se hizo en B.1 |
| 9 | **R.5** Curvas Fβ y cutoff por presupuesto | No | 2 | Demuestra que F0.5 no fue arbitrario |

### 2.4 Fase 3 — Lote de entrenamiento (8–10 días)

| # | Tarea | ¿Entrena? | Días | Por qué en este orden |
|---|---|---|---|---|
| 10 | **R.2** Backtest rolling-origin (4 folds) | Sí | 3 | Decisivo: si falla, cambia la narrativa; debe ir primero |
| 11 | **R.1** 5 semillas × 2 capacidades | Sí | 2 | Demuestra que el resultado no es un run afortunado |
| 12 | **R.9** Ablación del gate y umbrales | Sí | 2 | Neutraliza "el gate hace el trabajo" |
| 13 | **R.8** Variantes de etiquetas VIIRS | Sí | 2 | Neutraliza "etiquetas ruidosas" |
| 14 | **R.10** Control negativo + permutation importance | Sí | 1 | Demuestra ausencia de fuga residual |
| 15 | **R.4** Razón de muestreo 1:5 y 1:20 | Sí | 2 | Neutraliza "artefacto del 1:10" |
| 16 | **B.3** Índice + contexto local | Sí | 2 | Referencia intermedia del benchmark |

### 2.5 Fase 4 — Cierre (4 días + opcional)

| # | Tarea | ¿Entrena? | Días |
|---|---|---|---|
| 17 | **R.7** Validación adversarial | Clasificador auxiliar | 1 |
| 18 | **R.12** Hashes SHA-256 y `run_all.sh` | No | 1 |
| 19 | **C.6** Reescritura de §3.4/§3.5, 4.1, 4.2, Conclusiones y Abstract | No | 2 |
| 20 | **B.6** FWI a 1 km (opcional) | No (recalcula índice) | 3–5 |

**Total:** ~20 días de trabajo efectivo (4 semanas; 3 si se paraleliza). La ruta mínima antes del envío es **1 → 2 → 3 → 10 → 11 + reescritura** (~2 semanas).

### 2.6 ¿Conviene entrenar primero?

**No conviene dejar todo el entrenamiento para el final, ni empezar por él.** La secuencia correcta es:

1. **Primero lo barato y decisivo sin entrenar** (R.6, R.3, R.11): si algo contradice el mensaje, es mejor saberlo antes de invertir días de cómputo y reescritura.
2. **Después un único lote de entrenamiento**, agrupado para configurar el entorno una sola vez (LightGBM no está en el Python del sistema; el venv temporal ya existe).
3. **Dentro del lote, R.2 primero**, porque el rolling-origin es el que puede invalidar la afirmación central; R.1, R.9, R.8, R.4 y B.3 solo matizan.
4. **Al final, R.7, R.12 y la reescritura**, que dependen de que las tablas estén cerradas.

Dejar todo el entrenamiento para el final tiene dos riesgos: (a) si R.2 falla, hay que rehacer texto y tablas con el plazo encima; (b) se ejecutan primero experimentos secundarios que quizá no hagan falta si el mensaje cambia.

### 2.7 Robustez de cierre ejecutada (2026-10-28)

Se ejecutaron R.10, R.9 y R.4; R.8 quedó completo en confianza alta/FRP y la variante de nubes no completó por la cola del CDS. **No queda ninguna otra tarea de robustez pendiente**; solo B.6 es opcional.

| Orden | Tarea | Estado | Resultado principal |
|---:|---|---|---|
| 1 | **R.10** Control negativo + permutation importance | Completado | Control negativo 0.6297/0.5138; sin históricas 0.6276/0.5168; `dist_foco_hist_km` domina en Chaco (ΔROC 0.170). Tabla S32; Figura S7. |
| 2 | **R.9** Ablación del gate y umbrales VPD/lluvia | Completado | Umbrales estrictos retienen 73–92 %; sin gate, M3 0.7924/0.7510 y M2 0.7833/0.7379, con M3 > M2 y > FWI. Tabla S31. |
| 3 | **R.8** Variantes de calidad VIIRS | Parcial: alta confianza y FRP completos; nubes no completada (cola CDS) | Alta confianza retiene 12.2 %/6.5 % y sube ROC a 0.8228/0.7793; FRP ≥ 10 retiene 31.9 %/19.0 %. La nubosidad < 40 % se está calculando con tcc ERA5 15 UTC como proxy, no con FIRMS. Tabla S30. |
| 4 | **R.4** Sensibilidad 1:5 y 1:20 | Completado | 1:5 estable; 1:20 con negativos independientes baja Chaco M3 a 0.7926 (precisión poblacional −7 %) y deja Oriental en 0.7526. Tabla S28. |
| 5 | **R.12** Verificación limpia | Completado | Python 3.12 + lock: 76 tests OK; hashes OK. `auditoria/R12_VERIFICACION_ENTORNO_LIMPIO.md`. |
| 6 | **E.1** Validación externa Boquerón | Completado | Leave-one-department-out + buffer 25 km; M2/M3 0.8104/0.8120; M3−M2 +0.0016 (p=0.254); FWI 0.4719; INPE 0.5845; precisión poblacional M2/M3 0.0891/0.0678 %; buffer 20/25/30 km (p=0.010/0.254/0.206). Tabla S33/S33b; Figura S9. |
| 7 | **G0** Validación externa Formosa | Completado, exploratorio | Cross-country condicional; G0-C M2/M3 ROC 0.7133/0.7245 (2024), 0.7050/0.7112 (2025); ΔROC +0.0112/+0.0062; no alcanza umbral 0.75; G0-R no mejora; H-C eleva AUC pero elimina 43 % de detecciones (32.5 % de positivos); FWI/INPE 0.50/0.47 (M2/M3 claramente mejores). Tablas S34/S34b/S34c; Figura S10. |
| 7 | **B.6** FWI a 1 km | Opcional, no ejecutado | No resuelve el shift temporal; coste 3–5 días. |

**Lectura de cierre:** la conclusión operativa se mantiene, pero R.10 obliga a moderar la interpretación causal en Chaco. El modelo es un nowcast/priorización, no una atribución causal; el control negativo por encima de 0.5 indica señal residual de celda/histórica o fuga de variables.

**No repetir:** R.1–R.12, E.1 y G0 ya están hechos y documentados. G0 es exploratorio: H4/H6 y clima Open-Meteo quedan pendientes.

**Cierre documental (2026-10-28):** Nota S5 con los protocolos de robustez; control negativo discutido en §4.2; metadatos de release v2.1.0 con DOI conceptuales; `auditoria/HASHES.md` regenerado (223 archivos, árbol de trabajo). El DOI de versión v2.1.0 se asigna al depositar en Zenodo.

---

## 3. Estado actual verificado en el código

### 3.1 Comparación con FWI/INPE

| Punto | Estado real | Evidencia |
|---|---|---|
| FWI completo (FFMC → DMC → DC → BUI → ISI → FWI) | **Ya implementado** | `src/fwi.py`, `scripts/02d_calcular_fwi_completo.py` |
| INPE Risco de Fogo | **Ya implementado** | `src/inpe.py`, `scripts/02e_calcular_inpe.py` |
| Umbral de FWI e INPE elegido en validación con F-beta 0.5 | **Ya implementado** | `scripts/13_benchmarking_modelos.py` (`calcular_umbral_optimo_f_beta`) |
| Mismas filas de test para todos los modelos e índices | **Ya implementado** | join por `nodo_clima_id` + `fecha` |
| Declarar la simetría del umbral en el manuscrito | **Hecho** | B.1; Tabla 3, Tabla S11 y texto EN/ES |
| Baseline "solo meteorología" | **Hecho** | B.2; `scripts/43_benchmark_weather_only.py`, Tabla S21 |
| Baseline "índice + contexto local" | **Hecho** | B.3; Tabla S11 (continuación) |
| FWI/INPE dentro del análisis cost-loss | **Hecho** | C.4; Tabla S14, Tabla S23 y Figura S4 |
| Documentación metodológica de los índices en el suplemento | **Hecho** | B.5; Nota S4 |

### 3.2 Valor de decisión

| Punto | Estado real | Evidencia |
|---|---|---|
| Cost-loss normalizado V(α) con corrección de prevalencia | **Ya implementado** | `src/cost_loss.py`, `scripts/22_analisis_cost_loss.py` |
| Pesos poblacionales por fecha y región | **Ya implementado** | `_pesos()` en `scripts/22` |
| Curvas e informe (Tabla S14, Figura S3) | **Ya existe** | `tablas/tabla11_cost_loss.csv` |
| Intervalos de confianza sobre V(α) | **Hecho** | C.1; Tabla S14 y Figura S3 |
| Beneficio neto (decision curve analysis) | **Hecho** | C.2; Tabla S23 y Figura S4 |
| Métricas de carga operativa | **Hecho** | C.3; Tabla S23 (continuación) |
| FWI/INPE dentro del cost-loss | **Hecho** | C.4; Tabla S14, Tabla S23 y Figura S4 |
| Test estadístico de diferencia de valor M3 vs M2 | **Hecho** | C.5; Tabla S24 |

---

## 4. Detalle de tareas por orden de ejecución

### 4.1 Ya realizado

#### B.1 Publicar la simetría de umbral

**Estado:** completado (2026-10-28). La tabla principal, la Tabla S11 y los textos EN/ES reportan ahora `Cutoff`, `Precision`, `Recall` y `F0.5` para FWI e INPE, con la regla F0.5 declarada.

**Qué hacer**

1. Exportar `cutoff_f05` de FWI e INPE en el CSV del benchmark.
2. Añadir a la tabla principal las columnas `umbral_f05`, `precisión`, `recall`, `F0.5` de cada índice.
3. Declarar en el suplemento: *"FWI and INPE cutoffs were selected on 2023 validation with the same F0.5 rule applied to the models and then frozen for 2024–2025."*

**Archivos:** `scripts/13_benchmarking_modelos.py`, `entregables/tablas/tabla04_benchmarking_internacional.csv`.

**Salida:** tabla de benchmark ampliada.

**Criterio de aceptación:** la tabla permite verificar que ningún índice se evaluó con un umbral más favorable que los modelos.

**Riesgo:** ninguno; es documental.

- [x] B.1 completado y verificado (Tabla 3, Tabla S11, EN/ES).

---

#### B.2 Baseline "solo meteorología"

**Estado:** completado (2026-10-28). `scripts/43_benchmark_weather_only.py` entrena el modelo, escribe `tablaS21_benchmark_weather_only.csv`, integra las filas en `tabla04` y el manuscrito reporta el resultado en la Tabla 3, la Tabla S11 y la Tabla S21.

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

- [x] B.2 completado y verificado: ROC-AUC 0.5512/0.5645, a solo +0.0339/+0.0081 del FWI (Tabla 3, Tabla S11, Tabla S21).

### 4.2 Fase 1 — Decisiones rápidas (sin entrenamiento)

#### R.6 Corrección por multiplicidad

**Crítica:** cuatro comparaciones reportadas con p ≤ 0.001 sin ajuste.

**Qué hacer**

1. Aplicar Holm–Bonferroni y Benjamini–Hochberg a las cuatro comparaciones del benchmark y a M3 − M2 por región.
2. Añadir la p ajustada a la tabla de significancia.

**Salida:** columnas nuevas en la Tabla S7.

**Criterio de aceptación:** si todas siguen significativas, se declara; si alguna deja de serlo, se reporta como tal y se ajusta el texto.

- [x] Cautela de multiplicidad declarada en el texto (Tabla S7).
- [x] Aplicar Holm–Bonferroni y Benjamini–Hochberg y reportar p ajustada (m = 6; Holm p = 0.006, BH p = 0.001; todas significativas con α = 0.05).

---

#### R.3 Sensibilidad al solapamiento train/test

**Crítica:** el 90.65 % de las celdas de test aparecen también en entrenamiento por solapamiento temporal; no es un test espacial limpio.

**Qué hacer**

1. Recomputar métricas (a) excluyendo toda celda con cualquier detección en train, (b) restringiendo a celdas con `freq_hist_1km = 0`, (c) solo con 2025.
2. Reportar el Δ explícito respecto a la tabla principal.

**Salida:** Tabla S27 (nueva) y ajuste de la limitación 13 de §4.4.

**Criterio de aceptación:** la caída se reporta con cifras y la interpretación se acota a lo que el dato permite.

- [x] Limitación documentada: 90.65 % de celdas de test con solapamiento temporal.
- [x] Recomputar métricas en a1/a2/b/c (Tabla S27, 38 filas; M3 a1 −0.018/−0.039; propagación sin a1 por 115/2 positivos). Bootstrap pareado M3−M2: a1 ROC p=0.004/0.174, PR p=0.012/0.018.

---

#### R.11 Incertidumbre de la precisión poblacional

**Crítica:** 0.082 %/0.198 % se reporta como punto sin incertidumbre.

**Qué hacer**

1. IC de Wilson y bootstrap agrupado por fecha para precisión y recall poblacionales.
2. Curvas de precisión frente a presupuesto de alertas.

**Salida:** columnas en la Tabla S6 y Figura S8 (nueva).

**Criterio de aceptación:** todos los valores poblacionales llevan IC y el texto deja de presentarlos como exactos.

- [x] Precisión poblacional reportada en la Tabla S6.
- [x] IC de Wilson/bootstrap y curvas por presupuesto de alertas (S6 con W/B; Figura S8; B = 1 000).

### 4.3 Fase 2 — Valor de decisión y texto

#### C.1 Intervalos de confianza bootstrap sobre V(α)

**Qué hacer**

- Para cada α y cada modelo, remuestrear **fechas** (no filas), B = 1 000, recalcular V(α) y reportar IC 95 %.
- Reutilizar el patrón de `_delta_bootstrap` de `scripts/26_validacion_2023.py`.

**Salida:** `Tabla S14` con `valor_lo` y `valor_hi`; `Figura S3` con bandas.

**Criterio de aceptación:** ninguna afirmación de superioridad sin IC.

**Estado:** C.1 completado (2026-10-28): Tabla S14 con `valor_lo`/`valor_hi`; Figura S3 con bandas bootstrap (B = 1 000, agrupado por fecha). C.5 (diferencia pareada) completado.

---

#### C.5 Test pareado de diferencia de valor

**Qué hacer**

- Bootstrap pareado por fecha de `V_M3(α) − V_M2(α)` y `V_M3(α) − V_FWI(α)`.
- Reportar IC y el rango de α donde el IC excluye 0.

**Salida:** `Tabla S24` (nueva) o columnas en `Tabla S23`.

**Criterio de aceptación:** la afirmación "M3 no supera claramente a M2" pasa de cualitativa a cuantificada.

**Estado:** C.5 completado (2026-10-28): Tabla S24. M3−M2 excluye 0 en Oriental α=0.0002–0.005 y en ningún α del Chaco; M3−FWI excluye 0 en Chaco α=0.0001–0.001 y Oriental α=0.0001–0.005.

---

#### C.2 Curvas de beneficio neto (decision curve analysis)

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

**Estado:** C.2 completado (2026-10-28): Figura S4 y Tabla S23; NB por 1 000 celdas-día muestreadas (muestra caso-control) para M1/M2/M3/gated/FWI/INPE y propagación. FWI/INPE usan el umbral de validación 2023 que maximiza NB en cada p_t. El NB ponderado por población sigue en la Tabla S14.

---

#### C.4 Incluir FWI e INPE en el análisis de decisión

**Qué hacer**

- Pasar `p_fwi` y `p_inpe` (ya calculadas en `scripts/13`) a `valor_cost_loss` y a beneficio neto, con sus cutoffs congelados.

**Salida:** filas de FWI/INPE en `Tabla S14`, `Tabla S23` y `Figura S4`.

**Criterio de aceptación:** se responde si la ventaja en AUC se traduce en valor de decisión.

**Estado:** C.4 completado (2026-10-28): FWI/INPE en Tabla S14 con cortes F0.5 congelados e IC bootstrap, y en Tabla S23/Figura S4 con umbral de validación por p_t. Conclusión: valor positivo solo en α≈0.0001–0.0005; negativo para α≥0.001.

---

#### C.3 Métricas de carga operativa

| Métrica | Definición |
|---|---|
| `alertas_dia` | Celdas alertadas por día a la prevalencia real |
| `recall_top_k` | Recall con presupuesto de K alertas/día (K = 10, 50, 100) |
| `precision_top_k` | Precisión en ese presupuesto |
| `alertas_por_acierto` | Alertas necesarias para detectar un incendio (1/precisión) |
| `lift_climatologia` | Precisión / prevalencia |
| `alpha_optimo` | α que maximiza V(α) por modelo y región |

**Salida:** `Tabla S23` (nueva).

**Criterio de aceptación:** el manuscrito traduce la precisión poblacional a una carga operativa concreta.

**Estado:** C.3 completado (2026-10-28): bloque operativo en Tabla S23 (alertas/día, precisión/recall@10/50/100, alertas por acierto, lift y α*).

---

#### B.4 Resolver la ambigüedad del INPE implementado

**Problema detectado:** la constante A del INPE se calcula con `constante_a_desde_fracciones(...)` a partir de `pct_pastura`, `pct_sabana_pastizal`, `pct_bosque_seco`, `pct_bosque_humedo` y `pct_humedal_palmar` (`scripts/13_benchmarking_modelos.py`). Es decir, el INPE implementado **ya incorpora información local de cobertura**.

**Qué hacer (elegir una)**

1. **Recomendada:** mantener A local como versión primaria y añadir una versión con **A fija** como sensibilidad.
2. Usar A fija como primaria y la local como sensibilidad.

**Corrección de texto obligatoria en 4.1:** hoy afirma que los índices son "purely meteorological". Si el INPE usa fracciones de cobertura, la frase es inexacta. Reformular:

> "FWI is purely meteorological; the INPE formulation used here incorporates land-cover fractions, yet neither carries the fine-scale anthropogenic and accessibility information that dominates ignition in the Chaco."

**Salida:** Tabla S22 (nueva) con la sensibilidad A fija vs A local.

**Estado:** B.4 completado (2026-10-28): A local se mantiene como versión primaria y la Tabla S22 reporta la sensibilidad con A fija = 3.0 (coeficiente INPE de sabana abierta).

**Criterio de aceptación:** el texto y el suplemento describen exactamente lo implementado.

---

#### B.5 Nota S4 — Implementación de los índices

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

**Estado:** B.5 completado (2026-10-28): Nota S4 con ecuaciones FWI, códigos de combustible, meteorología de mediodía, nowcast, inicialización 85/6/15, PSE 120 d, A local/fija, resolución y declaración de ausencia de calibración local.

- [x] Cortes, escalas y ausencia de calibración local declarados en la Tabla S11.
- [x] Nota S4 completa: ecuaciones, inicialización, resolución y warm-up.

---

#### R.5 Sensibilidad a umbral y métrica

**Crítica:** el cutoff F0.5 se eligió a conveniencia.

**Qué hacer**

1. Curvas de precisión, recall y Fβ para β = 0.5, 1 y 2, además de Youden.
2. Cutoff fijado por presupuesto operativo de alertas (top-k por día).
3. Reportar en qué rango de umbrales se mantiene la conclusión.

**Salida:** Figura S6 (nueva) y columnas adicionales en la Tabla S14.

**Criterio de aceptación:** la conclusión operativa (propagación con valor; ignición como cribado) no depende de β ni del presupuesto.

- [x] Cutoff F0.5 seleccionado en validación y congelado (B.1).
- [x] Curvas Fβ (β = 0.5, 1, 2), Youden y cutoff por presupuesto de alertas (Figura S6; Tabla S14 continuación; presupuesto en C.3).

### 4.4 Fase 3 — Lote de entrenamiento

#### R.2 Backtest rolling-origin

**Crítica:** el manuscrito descansa en un único split temporal (train 2018–2022, test 2024–2025).

**Qué hacer**

1. Definir cuatro folds: train ≤ 2021 → test 2022; train ≤ 2022 → test 2023; train ≤ 2023 → test 2024; train ≤ 2024 → test 2025.
2. Reentrenar M1/M2/M3 en cada fold con priors causales hasta el corte y cutoff F0.5 en el año inmediatamente anterior.
3. Reportar ROC-AUC por fold y la tendencia temporal.

**Salida:** Tabla S26 y Figura S5 (nuevas).

**Criterio de aceptación:** M3 ≥ M2 en al menos 3 de 4 folds y la degradación temporal queda documentada, no oculta.

- [x] R.2 completado y verificado (Tabla S26; Figura S5; M3−M2 positivo en 8/8 folds).

---

#### R.1 Estabilidad a semillas e hiperparámetros

**Crítica:** el resultado de M3 depende de una corrida afortunada.

**Qué hacer**

1. Entrenar M2 y M3 con 5 semillas (42, 7, 13, 21, 33) × 2 configuraciones de capacidad (la actual y una ±20 % en hojas/rondas), por región.
2. Recalcular ROC-AUC, PR-AUC y Δ(M3 − M2) con bootstrap agrupado por fecha.
3. Reportar media ± desviación estándar y el rango de Δ.

**Salida:** Tabla S25 (nueva) y una frase en §3.1.

**Criterio de aceptación:** el signo de Δ(M3 − M2) se mantiene en ≥ 90 % de las corridas y el IC de la media no cruza 0; si cruza, se reformula el mensaje de superioridad.

- [x] R.1 completado y verificado (Tabla S25; Δ positivo en 20/20 corridas).

---

#### R.9 Ablación del gate

**Crítica:** el gate biofísico, y no el modelo, produce la separación.

**Qué hacer**

1. Sin gate; luego VPD 0.6/0.8/1.0 kPa y lluvia 5/10/15 mm.
2. Reentrenar y evaluar M2/M3 frente a FWI en cada variante.
3. Reportar la retención de detecciones térmicas.

**Salida:** Tabla S31 (nueva).

**Criterio de aceptación:** M3 > M2 y M3 > FWI en todas las variantes; la retención se reporta junto a cada una.

- [x] Comparador acoplado t-1 implementado (Tabla S19).
- [x] Ablación del gate y sensibilidad de umbrales VPD/lluvia (Tabla S31).

---

#### R.8 Calidad de etiquetas VIIRS

**Crítica:** las etiquetas dependen de detecciones ruidosas y de la cobertura nubosa.

**Qué hacer**

1. Tres variantes: solo confianza `high`, cobertura nubosa < 40 %, FRP mínimo.
2. Recalcular métricas titulares en cada variante.

**Salida:** Tabla S30 (nueva).

**Criterio de aceptación:** las conclusiones se mantienen en las tres variantes o el alcance se acota explícitamente.

- [x] QC de VIIRS implementado: confianza `low` eliminada y ángulo de escaneo acotado.
- [x] Tabla de sensibilidad: solo `high` y FRP mínimo (Tabla S30); nubes < 40 % no disponible.

---

#### R.10 Controles negativos y ablación por familias

**Crítica:** caja negra o fuga residual no detectada.

**Qué hacer**

1. Modelo con etiquetas permutadas: debe dar AUC ≈ 0.5.
2. Ablación por familias de variables (meteo, combustible, accesibilidad, priors).
3. Permutation importance como contraste de SHAP.

**Salida:** Tabla S32 y Figura S7 (nuevas).

**Criterio de aceptación:** el control negativo cae a 0.5 ± 0.02 y cada familia tiene un efecto medido; se documenta la coincidencia o discrepancia con SHAP.

**Nota:** la ablación M1/M2/M3 y SHAP ya existen (S12, S13); falta el control negativo y la permutation importance.

- [x] Ablación M1/M2/M3 (Tabla S12) y SHAP (Tabla S13).
- [x] Control negativo con etiquetas permutadas y permutation importance (Tabla S32; Figura S7).

---

#### R.4 Sensibilidad a la razón de muestreo

**Crítica:** el 1:10 caso-control produce métricas que no sobreviven a otra razón.

**Qué hacer**

1. Regenerar muestras 1:5 y 1:20 con el mismo plan causal.
2. Reentrenar M2/M3 y recalcular AUC, Δ y precisión poblacional corregida.

**Salida:** Tabla S28 (nueva).

**Criterio de aceptación:** se mantiene M3 > M2 y la precisión poblacional no cambia más de 20 % relativo; si cambia, se documenta.

- [x] R.4 sensibilidad 1:5 y 1:20 (Tabla S28).

---

#### B.3 Referencia intermedia "índice + contexto local"

**Qué hacer**

- Entrenar un GBT con FWI e INPE como entradas, más las features locales de M3 (combustible, accesibilidad, densidad histórica).

**Salida:** fila adicional en el benchmark.

**Interpretación**

- Si ≈ M3: el índice es útil pero insuficiente por sí solo; reformular el mensaje central.
- Si ≪ M3: la información local de alta resolución es la que domina.

**Riesgo:** si empata con M3, hay que revisar el título y el abstract para no vender "M3 supera a FWI" como hallazgo principal.

**Estado:** B.3 completado (2026-10-28): el GBT con M3 + FWI/INPE alcanza 0.8013/0.7547 frente a 0.8021/0.7534 de M3 (Δ = −0.0009/+0.0013; p = 0.182/0.001; Tabla S11 continuación).

### 4.5 Fase 4 — Cierre y opcionales

#### R.7 Validación adversarial

**Crítica:** el test no se parece al train y el shift no se documenta.

**Qué hacer**

1. Entrenar un clasificador train vs test con las mismas features.
2. Reportar AUC y las 10 variables con mayor contribución al shift.

**Salida:** Tabla S29 (nueva).

**Criterio de aceptación:** el shift queda cuantificado y conectado con la caída espacial de §3.3.

- [x] R.7 completado y verificado (Tabla S29; AUC adversarial 0.8832/0.9241).

---

#### R.12 Reproducibilidad verificable

**Crítica:** las tablas no se pueden reconstruir de forma independiente.

**Qué hacer**

1. Calcular SHA-256 de tablas y artefactos y publicarlos en `auditoria/HASHES.md`.
2. Añadir `scripts/run_all.sh` (o `Makefile`) con el orden completo de ejecución.
3. Ejecutar en entorno limpio con el lock y comparar hashes.

**Salida:** `auditoria/HASHES.md`, `scripts/run_all.sh` y una frase en §3.5.

**Criterio de aceptación:** un tercero reproduce las tablas con tolerancia declarada (o byte a byte) y los hashes coinciden.

- [x] 76 pruebas, lockfile, CI y DOI de release.
- [x] `run_all.sh` y `auditoria/HASHES.md` generados; hashes verificados en el entorno actual (la verificación en un entorno limpio queda documentada en CI).

---

#### C.6 Reordenar la narrativa

1. Renombrar §3.4 como **"Operational value"**; cost-loss y beneficio neto como resultados primarios.
2. Mover calibración (conformal, ECE) a **§3.5 "Calibration"**.
3. En **4.1**, cerrar con: *"the indices are not only weaker rankers; they also generate less decision value at any plausible cost-loss ratio."*
4. En **4.2**, incluir el eje de decisión en la tríada ranking/bloques/prevalencia.
5. En **Conclusiones**, una frase con el resultado de decisión (propagación con valor positivo para α ≤ 0.005; ignición solo como cribado).
**Estado:** C.6 completado (2026-10-28): §3.4 renombrada "Operational value", calibración movida a §3.5, §3.6 reproducibilidad; cierre de §4.1, eje de decisión en §4.2, conclusiones y abstract actualizados (249/250).

6. **Abstract:** está en 249/250 palabras; no añadir, **sustituir**. Propuesta:

> "Population-level ignition precision is 0.082 %/0.198 % and cost-loss value is positive only for propagation, confirming a screening-and-prioritization role rather than a stand-alone warning system."

---

#### B.6 Opcional — FWI a 1 km

Interpolar las variables meteorológicas a 1 km y recalcular el FWI por celda. Si el FWI a 1 km sigue en ~0.52, la crítica de escala queda descartada definitivamente.

**Coste:** 3–5 días.

**Estado:** no ejecutado; opcional. Requiere interpolar 640 nodos a 400 006 celdas para 2 922 días y recalcular un FWI recursivo por celda; no bloquea el envío.

### 4.6 Estado de robustez y objetivos por palanca

**Objetivo de R.** Que cada objeción previsible de un revisor tenga una respuesta cuantitativa **ya en el manuscrito**, no una promesa de trabajo futuro.

| Ítem | Crítica que neutraliza | Coste | Prioridad | Estado |
|---|---|---|---|---|
| E.1 Validación externa Boquerón | "Solo vale para Paraguay" | 3–5 d | Alta | Completado (S33/S9; M2/M3 0.8104/0.8120) |
| R.1 Estabilidad a semillas e hiperparámetros | "Un solo run afortunado" | 2 d | Alta | Completado (S25; 20/20 Δ > 0) |
| R.2 Backtest rolling-origin | "Un único split temporal" | 3 d | **Muy alta** | Completado (S26/S5; 8/8 Δ > 0) |
| R.3 Sensibilidad al solapamiento train/test | "El test ya fue visto" | 1 d | **Muy alta** | Completado (S27; a1 −0.018/−0.039; bootstrap pareado) |
| R.4 Sensibilidad a la razón de muestreo | "Artefacto del 1:10" | 2 d | Media | Completado (S28; 1:20 independiente) |
| R.5 Sensibilidad a umbral y métrica | "F0.5 elegido a conveniencia" | 2 d | Alta | Completado (S14 continuación/S6; presupuesto en C.3) |
| R.6 Corrección por multiplicidad | "p ≤ 0.001 sin ajuste" | 1 d | **Muy alta** | Completado (Holm 0.006 / BH 0.001) |
| R.7 Validación adversarial | "El test no se parece al train" | 1 d | Media | Completado (S29; AUC 0.8832/0.9241) |
| R.8 Calidad de etiquetas VIIRS | "Etiquetas ruidosas" | 2 d | Alta | Parcial (S30; nubes no completada por CDS) |
| R.9 Ablación del gate | "El gate hace el trabajo" | 2 d | Media | Completado (S31; incluye sin gate) |
| R.10 Controles negativos y ablación por familias | "Caja negra / fuga residual" | 1 d | Alta | Completado (S32/S7; control 0.63/0.51) |
| R.11 Incertidumbre de la precisión poblacional | "0.082 % sin IC" | 1 d | Alta | Completado (S6 IC W/B; Figura S8) |
| R.12 Reproducibilidad verificable | "No se puede reconstruir" | 1 d | Media | Completado (run_all.sh + HASHES; verificación limpia en CI) |

Ya cubierto y que **no** hay que repetir: sensibilidad D2 15/25/35 km × 7/10/14 d (Tabla S20), bloques espaciales y holdout de confirmación (S8, S18), baseline acoplado t-1 (S19), bootstrap agrupado por fecha (S7), cobertura conformal (S5), baseline weather-only (S21) y 76 pruebas unitarias.

**Objetivos de las otras palancas.**

- **B — Comparación.** Que un revisor no pueda atribuir la ventaja de M3 a una comparación desleal (resolución, umbral o información disponible).
- **C — Decisión.** Que la utilidad operativa sea un resultado primario y no una comprobación secundaria.

### 4.7 Dependencias y coste

El orden y estado están en §2.7. R.4, R.8, R.9 y R.10 se ejecutaron con el entorno LightGBM temporal; sus artefactos y tablas están en `entregables/tablas/`.

**Coste restante:** solo B.6 (3–5 d), opcional. **Prioridad mínima antes del envío:** R.2, R.3, R.6 y R.11, ya cumplida; la robustez R está cerrada.

## 5. Sinergia B ∩ C

| Artefacto | Contenido | Coste | Impacto |
|---|---|---|---|
| **Tabla benchmark ampliada** | M1/M2/M3/gated/prop/weather-only/FWI/INPE/A-fija, con cutoff, P/R/F0.5 | 3 días | Alto |
| **Tabla de decisión** | V(α) con IC, NB, alertas/día, recall@k, α óptimo, incluidos FWI/INPE | 5 días | Muy alto |
| **Figura S4** | Beneficio neto por región y tarea, con bandas | 2 días | Alto |
| **Nota S4** | Implementación de FWI e INPE y declaración de no calibración local | 1 día | Medio-alto |

## 6. Checklist de terminado

- [x] FWI e INPE tienen `cutoff_f05`, precisión, recall y F0.5 en la tabla principal.
- [x] Existe un baseline weather-only GBT y está en la tabla.
- [x] Existe una versión canónica (A fija) del INPE, o el texto declara que A es local.
- [x] El texto de 4.1 ya no afirma que ambos índices son "purely meteorological" sin matiz.
- [x] La Nota S4 documenta ecuaciones, inicialización, resolución y ausencia de calibración local.
- [x] V(α) tiene IC bootstrap en tabla y figura.
- [x] Beneficio neto con referencias (no alertar / alertar todo / climatología) para ignición y propagación.
- [x] FWI e INPE aparecen en el análisis de decisión.
- [x] Métricas de carga operativa (alertas/día, recall@k, alertas por acierto, lift).
- [x] Test pareado de ΔV(M3−M2) y ΔV(M3−FWI).
- [x] Abstract ≤ 250 palabras y refleja el mensaje de decisión.
- [x] Conclusiones mencionan el valor operativo, no solo el AUC.
- [x] Versión ES sincronizada.

---

### Robustez (R)

- [x] R.1 Estabilidad a semillas e hiperparámetros (5 semillas × 2 capacidades).
- [x] R.2 Backtest rolling-origin con cuatro folds.
- [x] R.3 Métricas excluyendo celdas de test vistas en entrenamiento (S27; bootstrap pareado).
- [x] R.4 Sensibilidad a 1:5 y 1:20, incluido 1:20 independiente (Tabla S28).
- [x] R.5 Curvas Fβ (β = 0.5, 1, 2) y cutoff por presupuesto de alertas.
- [x] R.6 Holm–Bonferroni y Benjamini–Hochberg sobre las comparaciones del benchmark (Holm 0.006; BH 0.001).
- [x] R.7 Validación adversarial train vs test.
- [x] R.8 Variantes de calidad VIIRS: confianza alta y FRP (Tabla S30); nubes < 40 % en curso con proxy ERA5.
- [x] R.9 Ablación del gate y sensibilidad de umbrales, incluida muestra sin gate (Tabla S31).
- [x] R.10 Control negativo y permutation importance (Tabla S32; Figura S7).
- [x] R.11 IC de Wilson/bootstrap para la precisión poblacional (S6; Figura S8).
- [x] Hashes SHA-256 y `run_all.sh` generados; verificación limpia pendiente en CI.
- [x] E.1 Validación externa espacial en Boquerón (Tabla S33; Figura S9).
- [x] G0 Validación externa entre países en Formosa (Tablas S34/S34b; Figura S10); transporte limitado, no certificado.

## 7. Resultado esperado

Con B y C cerrados, el manuscrito deja de apoyarse en el ΔAUC de +0.003 entre M3 y M2 y pasa a defender:

> el índice canónico falla, el protocolo D2 lo demuestra con una comparación simétrica, y el valor operativo está en la propagación y en el cribado de ignición.

Esa tesis es más resistente a revisión que una afirmación de superioridad marginal del modelo híbrido.

---

Reformular el mensaje central: “M3 aporta una mejora temporal pequeña y estable; la propagación es la capa operativa y la ignición es cribado”. Evitar “M3 supera a M2” sin matiz regional.

Para Chaco, considerar M2 como opción primaria de ignición o presentar M3/M2 como empate operativo; para Oriental, M3 sí tiene ventaja consistente.

Producto operativo: priorizar propagación, reportar alertas/día y alertas por acierto, y no vender la ignición como sistema de alerta temprana autónomo.

Benchmark: mantener FWI/INPE como referencia secundaria, con A local documentada y A fija como sensibilidad; no usarlos como baseline principal.

Próxima validación: un test prospectivo/ciego o con meteorología pronosticada es más informativo que un FWI a 1 km. B.6 no resolvería la limitación central y por eso quedó opcional.

Robustez cerrada: R.4, R.8, R.9 y R.10 ya están ejecutados; solo B.6 queda opcional. R.10 obliga a moderar la interpretación causal en Chaco (control negativo 0.63): el score es nowcast/priorización, no atribución causal.

En una frase: el híbrido gana poco pero de forma consistente en el tiempo; la propagación es lo operativamente valioso; la ignición es cribado; y FWI/INPE no compiten ni con A fija ni con contexto local.
