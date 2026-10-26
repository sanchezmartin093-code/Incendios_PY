# Estratificación del desempeño por distancia al foco t-1

Esta tabla NO es un análisis de sensibilidad al buffer de muestreo. Para evaluar el buffer hay que remuestrear y reentrenar con 10, 20 y 30 km. Aquí sólo se filtra el test por distancia al foco de ayer.

|   distancia_min_t1_km | condicion                                     |   n_muestras_chaco |   auc_ign_chaco |   auc_prop_chaco |   n_muestras_oriental |   auc_ign_oriental |
|----------------------:|:----------------------------------------------|-------------------:|----------------:|-----------------:|----------------------:|-------------------:|
|                    10 | Incluye celdas a >=10 km del foco t-1         |             695417 |          0.7984 |           0.8067 |                903310 |             0.7976 |
|                    20 | Incluye celdas a >=20 km del foco t-1         |             614560 |          0.7993 |           0.7929 |                659153 |             0.797  |
|                    30 | Sólo celdas a >=30 km del foco t-1 (alejadas) |             470316 |          0.8029 |           0.7982 |                428761 |             0.8016 |
