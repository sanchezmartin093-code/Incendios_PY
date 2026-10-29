# G0-Formosa: reporte de ejecución

## Estado

G0 se ejecutó como **validación externa cross-country condicional**. La armonización H-A es la primaria; H-B y H-C se reportan como sensibilidades. Los resultados son **exploratorios**: H4 (cross-tab con MapBiomas Chaco C5) y H6 (auditoría visual) no se completaron, y el test no se selló criptográficamente antes de la primera inspección agregada.

## Datos

- Límite: geoBoundaries ARG ADM1 (auditoría); área 75,229 km²; 75,236 celdas de 1 km².
- Clima: ERA5-Land CDS + swvl1 CDS; Open-Meteo incompleto por cuota horaria, por lo que se usó **CDS + ET0 Hargreaves** como fallback documentado. Esto difiere de la corrección Open-Meteo usada en Paraguay.
- Combustible: MapBiomas Argentina Colección 3, mapas 2017–2024; H-A.
- Hansen GFC-2025 v1.13; tiles 20S_070W y 20S_060W.
- VIIRS: SNPP + NOAA-20, filtros `scan<=1.5`, confianza no baja, `type=0`.
- Test: 68,959 filas caso-control 1:10; 6,269 positivos (2024: 3,429; 2025: 2,840). Priors congelados a 2023-12-01.

## Armonización

Composición 2024 (H-A): bosque 77.2 % (clase 6 inundable 22.3 %), humedal 8.8 %, sabana 5.5 %, pastura 2.4 %, agricultura 3.2 %, no combustible 1.2 %.

Diagnóstico de distribución Formosa vs Paraguay-Chaco:

- 19 de 44 features con PSI > 0.2 o KS > 0.1.
- Mayores shifts: `pct_pastura_ganadera` PSI 1.18; `presion_quema_pastura` 1.03; `borde_bosque_agro_pct` 0.51; `indice_ecotono` 0.47; `temp_min_c` 0.25; `densidad_hist_25km` 0.24.
- La mediana de pastura en Formosa es 0 % frente a 0.19 % en Paraguay-Chaco; el modelo M3 depende de variables antropogénicas con soporte muy distinto.

Esto es una **extrapolación fuerte**, no una réplica.

## Resultados primarios G0-C

| Año | Modelo | ROC-AUC | PR-AUC | Brier | ECE | F0.5 | ΔROC M3−M2 (IC 95 %) |
|---|---|---:|---:|---:|---:|---:|---:|
| 2024 | M2 | 0.7133 | 0.1839 | 0.0788 | 0.0078 | 0.2005 | — |
| 2024 | M3 | 0.7245 | 0.1943 | 0.0783 | 0.0106 | 0.2061 | +0.0112 (0.0065–0.0155) |
| 2025 | M2 | 0.7050 | 0.1735 | 0.0793 | 0.0129 | 0.1881 | — |
| 2025 | M3 | 0.7112 | 0.1716 | 0.0794 | 0.0153 | 0.1551 | +0.0062 (−0.0001–0.0129) |
| pooled | M2 | 0.7094 | 0.1788 | 0.0790 | 0.0100 | 0.1957 | — |
| pooled | M3 | 0.7184 | 0.1836 | 0.0788 | 0.0125 | 0.1862 | +0.0090 (0.0050–0.0128) |

- **Transporte absoluto:** no alcanza el umbral pre-registrado de ROC-AUC > 0.75.
- **Ventaja híbrida:** positiva y clara en 2024; borderline en 2025 (p ≈ 0.05).
- **Calibración:** G0-C reduce ECE frente a G0-Z (0.0078–0.0153 vs 0.033–0.054), pero el cutoff local reduce el F0.5 de M3 en 2025.

## Brazos adicionales

| Brazo | 2024 M2/M3 ROC | 2025 M2/M3 ROC | Lectura |
|---|---:|---:|---|
| G0-Z (sin recalibrar) | 0.7133 / 0.7245 | 0.7050 / 0.7112 | mismo ranking; peor calibración |
| G0-C (recalibrado local) | 0.7133 / 0.7245 | 0.7050 / 0.7112 | mejor ECE; cutoff local |
| G0-R (reentrenado PY+Formosa) | 0.7178 / 0.7275 | 0.7144 / 0.7150 | mejora marginal; no resuelve 2025 |

## Comparación con modelos clásicos

FWI canadiense e INPE se recalcularon con el mismo clima de Formosa y las fracciones H-A; sus cortes se eligieron en 2023 maximizando F0.5.

| Año | FWI ROC / PR | INPE ROC / PR | M2 ROC / PR | M3 ROC / PR |
|---|---:|---:|---:|---:|
| 2024 | 0.498 / 0.093 | 0.466 / 0.084 | 0.713 / 0.184 | 0.725 / 0.194 |
| 2025 | 0.520 / 0.096 | 0.464 / 0.084 | 0.705 / 0.173 | 0.711 / 0.172 |

M2/M3 superan claramente a FWI/INPE en discriminación y F0.5. El INPE logra recall alto solo con precisión ≈0.09; el FWI queda cerca del azar. Aun así, M2/M3 no alcanzan el umbral de transporte de 0.75. La comparación clásica no rescata la validación entre países: solo confirma que los modelos entrenados son el mejor ranking disponible en Formosa.

## Sensibilidades

- **H-B** (clase 6 → humedal): M3 2024 0.7256, 2025 0.7127. Prácticamente igual a H-A.
- **H-C** (excluir celdas con ≥50 % clase 6): M3 2024 0.7616, 2025 0.7357. Sube el AUC, pero elimina 43 % de las detecciones y 32.5 % de los positivos, y 22 % del área; no es el resultado primario.
- **Buffer t−1**: 20 km M3 0.7173/0.7041; 25 km 0.7245/0.7112; 30 km 0.7251/0.7111. El signo M3 > M2 se mantiene, pero el nivel absoluto no alcanza 0.75.
- **G0-R** no mejora materialmente sobre G0-C: agregar Formosa al entrenamiento no compensa el cambio de régimen.

## Interpretación

1. **El ranking se transporta parcialmente**, pero por debajo del umbral operativo pre-registrado.
2. **M3 conserva una ventaja pequeña** sobre M2 en 2024; en 2025 el IC incluye cero.
3. **La armonización de clases es el cuello de botella**: 22 % del área es bosque inundable y 43 % de los fuegos ocurren allí; el modelo no tiene una feature específica para ese combustible.
4. **La diferencia climática** (CDS/Hargreaves vs Open-Meteo) y el cambio de régimen de pastura limitan la comparación con Boquerón.
5. El resultado correcto es **transporte condicional/limitado**, no una validación exitosa ni una demostración de superioridad de M3.

## Próximos pasos obligatorios antes de publicación

- Completar H4 (cross-tab Argentina C3 vs Chaco C5) y H6 (auditoría visual).
- Repetir clima con Open-Meteo completo cuando la cuota lo permita; reportar como sensibilidad.
- Congelar/hashear protocolo y test; repetir la evaluación una sola vez.
- Decidir si se agrega una feature explícita `pct_bosque_inundable` y se reentrena (G0-R+), fuera del conjunto de 44 features.
