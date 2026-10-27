# Estratificación del desempeño por distancia al foco t-1

Esta tabla NO es un análisis de sensibilidad al buffer de muestreo. Para evaluar el buffer hay que remuestrear y reentrenar con 10, 20 y 30 km. Aquí sólo se filtra el test por distancia al foco de ayer.

|   distancia_min_t1_km | condicion                                     |   n_muestras_chaco |   auc_ign_chaco |   auc_prop_chaco |   n_muestras_oriental |   auc_ign_oriental |
|----------------------:|:----------------------------------------------|-------------------:|----------------:|-----------------:|----------------------:|-------------------:|
|                    10 | Incluye celdas a >=10 km del foco t-1         |             176616 |          0.8021 |           0.7536 |                199936 |             0.7534 |
|                    20 | Incluye celdas a >=20 km del foco t-1         |             176616 |          0.8021 |           0.7645 |                199936 |             0.7534 |
|                    30 | Sólo celdas a >=30 km del foco t-1 (alejadas) |             157696 |          0.803  |         nan      |                168654 |             0.7553 |
