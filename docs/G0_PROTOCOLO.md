# G0-Formosa: protocolo congelado de validación externa condicional

## Pregunta

¿El sistema M2/M3-Chaco entrenado en Paraguay transporta discriminación y calibración a Formosa (Argentina), y la ventaja M3 > M2 se mantiene fuera del país?

## Diseño

- **G0-Z (transporte puro):** boosters M2/M3-Chaco congelados; calibrador y cutoff de Paraguay; sin datos de Formosa en entrenamiento.
- **G0-C (primario):** mismos boosters; recalibración Platt y cutoff F0.5 usando solo Formosa 2023; test 2024 y 2025.
- **G0-R (secundario):** reentrenamiento M2/M3 con Paraguay + Formosa 2018–2022; validación 2023; test 2024–2025. No es validación externa.

## Splits

| Uso | Paraguay | Formosa |
|---|---|---|
| Entrenamiento G0-Z/G0-C | 2018–2022, excluyendo 25 km desde el límite de Formosa | — |
| Validación G0-Z/G0-C | 2023 | 2023 (solo G0-C) |
| Test primario | — | 2024 |
| Replicación | — | 2025 |
| Entrenamiento G0-R | 2018–2022 | 2018–2022 |
| Validación G0-R | 2023 | 2023 |

**Test sellado:** no se usan 2024–2025 para elegir variables, hiperparámetros, buffer, cutoff, mapeo de clases ni modelo. La auditoría de viabilidad revisó conteos agregados; no revisó desempeño M2/M3.

## Métrica primaria

ΔROC-AUC M3−M2 en Formosa 2024, buffer de 25 km, con bootstrap pareado agrupado por fecha (1 000 réplicas). 2025 es replicación; pooled es descriptivo. No se cambia el buffer primario.

## Armonización MapBiomas

MapBiomas Argentina Colección 3 (30 m) es la fuente primaria; MapBiomas Chaco Colección 5 (2023) es control. El crosswalk está en `auditoria/G0/crosswalk_mapbiomas_argentina.csv`.

- **H-A (primaria):** 3/4/6 → `pct_bosque_seco`; 9 → plantación; 11 → humedal; 10/12/63/66/77 → sabana; 15 → pastura; 18/19/36 → agricultura; 21 → pastura; 22/24/25/26/33/34 → no combustible; 73 → humedal; 27 → missing.
- **H-B (sensibilidad):** 6 → `pct_humedal_palmar`.
- **H-C (sensibilidad):** enmascarar celdas con ≥50 % de clase 6.

La clase 6 se asigna a bosque en H-A porque el modelo de Paraguay fue entrenado con la clase 6 mapeada a `bosque_seco` en el Chaco. El nombre “seco” es operativo, no una afirmación ecológica.

## Gates de ejecución

- **H1:** 100 % de códigos presentes tienen regla; desconocidos = 0.
- **H2:** fracciones no negativas; suma ≤100; sin doble conteo jerárquico.
- **H3:** sin defaults silenciosos de features; columnas requeridas completas.
- **H4:** cross-tab Argentina C3 vs Chaco C5 2023; discrepancias >10 pp investigadas.
- **H5:** PSI/KS de features Formosa vs Paraguay-Chaco; PSI >0.2 o KS >0.1 se reporta como extrapolación.
- **H6:** muestra visual de 300 puntos para clases 3, 6, 11, 19 y 21.
- **H7:** test sellado hasta que crosswalk, features y modelo estén hasheados.

Si H1–H3 fallan, G0-C no se ejecuta. Si H4–H6 fallan, se reporta y se decide antes del test.

## Sensibilidades obligatorias

1. Buffer 20/25/30 km (25 primario).
2. H-A/H-B/H-C.
3. Priors congelados a 2023 vs actualizados con 2024 para 2025.
4. Priors locales de Formosa vs priors importados de Paraguay.
5. MapBiomas Argentina C3 vs Chaco C5 2023.
6. Calibración Paraguay vs Formosa 2023.
7. Muestreo 1:5, 1:10, 1:20.
8. Formosa completa vs interior sin buffer de borde.

## Interpretación

- **Transporte:** M2 o M3 ROC-AUC > 0.75 y calibración aceptable.
- **Ventaja híbrida:** ΔROC > 0 con IC 95 % que excluye 0.
- **Si falla:** se reporta fallo; no se cambia el mapeo ni el buffer para obtener significancia.

## Archivos

- `auditoria/G0/`
- `datos/processed/formosa_g0/`
- scripts `69a`–`69f`
- Tabla S34, Figura S10, Nota S5.11
