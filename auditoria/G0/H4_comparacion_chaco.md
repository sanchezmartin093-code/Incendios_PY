# H4 — Comparación MapBiomas Argentina C3 vs MapBiomas Chaco C5 (2023)

## Alcance

Se comparó la composición oficial 2023 de **MapBiomas Chaco Colección 5** para Formosa con la composición agregada de **MapBiomas Argentina Colección 3** usada en G0 (mapa 2023, fila 2024).

Limitación: Chaco C5 está distribuido como asset de Google Earth Engine, no como GeoTIFF directo. No había credenciales GEE en el entorno, por lo que **no se pudo hacer la matriz de confusión pixel a pixel**. La comparación es agregada, con las estadísticas oficiales de C5.

## Resultados

| Clase | C5 2023 (%) | C3 2023 (%) | Diferencia (pp) |
|---|---:|---:|---:|
| Bosque (sin inundable) | 56.49 | 54.92 | −1.57 |
| Bosque inundable | 17.62 | 22.27 | +4.65 |
| Bosque total | 74.11 | 77.19 | +3.08 |
| Humedal | 13.99 | 8.83 | −5.16 |
| Sabana/pastizal | 2.78 | 5.48 | +2.69 |
| Pastura | 2.08 | 2.43 | +0.36 |
| Agricultura | 6.27 | 3.16 | −3.12 |
| Plantación | 0.02 | 0.00 | −0.02 |
| No vegetado/agua | 0.74 | 1.17 | +0.43 |

- Área C5 Formosa: **73,756 km²**.
- Área de la grilla G0: **75,229 km²** (diferencia 1.96 %, por borde y definición de polígono).
- Diferencia máxima absoluta: **5.16 pp** (humedal).
- Clase 6: **17.62 % vs 22.27 %** (+4.65 pp), dentro del umbral de 10 pp.

## Gate H4

Criterio predefinido: diferencias <10 pp en bosque, bosque inundable y humedal; kappa si hubiera raster.

- Diferencias de área: **PASS**.
- Kappa pixel a pixel: **NO EVALUABLE** por falta de raster C5.
- Resultado: **H4 parcial, consistente en agregado**.

La diferencia de clase 6 y humedal es la más relevante: C3 asigna más área a bosque inundable y menos a humedal que C5. Eso apoya la sensibilidad H-B/H-C, pero no invalida H-A a nivel agregado.

## Consecuencia

La armonización H-A no muestra una discrepancia agregada grande contra C5. Sin embargo, la falta de kappa pixel a pixel impide declarar H4 completamente cerrado. Para cerrarlo haría falta exportar C5 desde GEE a la grilla de 1 km y repetir la matriz de confusión.
