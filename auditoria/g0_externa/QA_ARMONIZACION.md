# G0-Formosa: QA de armonización

| Gate | Estado | Evidencia |
|---|---|---|
| H1 cobertura de códigos | PASS | Códigos presentes en Formosa: 3, 4, 6, 11, 12, 15, 19, 24, 25, 27, 33, 63, 66, 77; todos con regla |
| H2 fracciones | PASS | No negativas; suma media 98.25 % (0/27 no observados); sin doble conteo padre/hijo |
| H3 sin defaults silenciosos | PASS | `69f` falla si falta una feature o hay nulos; no se usó el fallback de `agregar_features_antropicas` |
| H4 cross-tab Argentina C3 vs Chaco C5 | PARCIAL PASS | Composición agregada: diferencia máxima 5.16 pp; clase 6 +4.65 pp. C5 está solo en GEE, no se hizo kappa pixel a pixel |
| H5 distribución de features | ALERTA | 19/44 con PSI > 0.2 o KS > 0.1; ver `distribucion_features_g0.csv` |
| H6 auditoría visual | PILOTO | 60/300 revisados; 53/60 consistentes (88.3 %); agricultura 8/12; falta completar 240 |
| H7 test sellado | PARCIAL | Protocolo escrito antes de métricas, pero sin hash/commit previo; conteos agregados ya conocidos |

## Distribución

Las features con mayor cambio son:

| Feature | PSI | KS | Mediana PY | Mediana Formosa |
|---|---:|---:|---:|---:|
| pct_pastura_ganadera | 1.18 | 0.37 | 0.19 | 0.00 |
| presion_quema_pastura | 1.03 | 0.37 | 0.003 | 0.000 |
| borde_bosque_agro_pct | 0.51 | 0.29 | 0.54 | 0.00 |
| indice_ecotono | 0.47 | 0.27 | 0.004 | 0.000 |
| temp_min_c | 0.25 | 0.20 | 21.86 | 18.54 |
| densidad_hist_25km | 0.24 | 0.19 | 530 | 834 |

## Composición

- Clase 6 (bosque inundable): 22.3 % del área y 42.9 % de las detecciones.
- Pastura: 2.4 % del área; mediana 0 % por celda.
- La clase 6 se mapea a `pct_bosque_seco` en H-A por consistencia con el entrenamiento de Paraguay; H-B la mueve a humedal y H-C la enmascara.

## Consecuencia

G0-C **no cumple el gate de transporte** (ROC > 0.75) y la armonización no es neutral. Los resultados se reportan como exploratorios y condicionales. Antes de publicación deben completarse H4/H6 y repetirse el clima con Open-Meteo.
