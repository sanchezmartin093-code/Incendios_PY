# Estratificación del desempeño por distancia al foco t-1

Esta tabla NO es un análisis de sensibilidad al buffer de muestreo. Para evaluar el buffer hay que remuestrear y reentrenar con 10, 20 y 30 km. Aquí sólo se filtra el test por distancia al foco de ayer.

|   distancia_min_t1_km | condicion                                     |   n_muestras_chaco |   auc_ign_chaco |   auc_prop_chaco |   n_muestras_oriental |   auc_ign_oriental |
|----------------------:|:----------------------------------------------|-------------------:|----------------:|-----------------:|----------------------:|-------------------:|
|                    10 | Incluye celdas a >=10 km del foco t-1         |             176616 |          0.7852 |           0.7331 |                199936 |             0.748  |
|                    20 | Incluye celdas a >=20 km del foco t-1         |             176616 |          0.7852 |           0.7445 |                199936 |             0.748  |
|                    30 | Sólo celdas a >=30 km del foco t-1 (alejadas) |             157663 |          0.7866 |         nan      |                168751 |             0.7495 |
