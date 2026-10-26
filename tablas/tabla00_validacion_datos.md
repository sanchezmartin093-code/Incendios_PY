# Validación de integridad de datos

| seccion       | chequeo                                            | valor      | estado   | nota                                                                                 |
|:--------------|:---------------------------------------------------|:-----------|:---------|:-------------------------------------------------------------------------------------|
| grilla        | filas                                              | 400006     | OK       |                                                                                      |
| grilla        | cell_id_unicos                                     | 400006     | OK       |                                                                                      |
| grilla        | celdas_Chaco                                       | 240880     | OK       |                                                                                      |
| grilla        | celdas_Oriental                                    | 159126     | OK       |                                                                                      |
| clima         | filas                                              | 1870080    | OK       |                                                                                      |
| clima         | nodos                                              | 640        | OK       |                                                                                      |
| clima         | fecha_min                                          | 2018-01-01 | OK       |                                                                                      |
| clima         | fecha_max                                          | 2025-12-31 | OK       |                                                                                      |
| clima         | precipitacion_mayor_300mm                          | 4115       | WARN     | valores posibles pero improbables; revisar desacumulación                            |
| clima         | precipitacion_max_mm                               | 1253.985   | WARN     | pendiente ejecutar scripts/02c_corregir_clima_openmeteo.py cuando haya cuota externa |
| clima         | et0_distinto_de_cero                               | 0          | WARN     |                                                                                      |
| clima         | rafaga_es_proxy_1.3x                               | 4e-06      | WARN     |                                                                                      |
| firms         | celda_dias_positivos                               | 959590     | OK       |                                                                                      |
| firms         | fecha_min                                          | 2018-01-01 | OK       |                                                                                      |
| firms         | fecha_max                                          | 2026-01-04 | WARN     | puede exceder la ventana de clima                                                    |
| firms_raw     | detecciones                                        | 3685836    | OK       |                                                                                      |
| firms_raw     | scan_max_km                                        | 0.8        | WARN     | si max_scan<=1.5 el filtro Bow-Tie no descarta registros                             |
| firms_raw     | detecciones_type_no_0                              | 5319       | WARN     | no se filtró type; revisar fuentes industriales/offshore                             |
| dataset       | filas                                              | 711087509  | OK       |                                                                                      |
| dataset       | fuegos                                             | 863074     | OK       |                                                                                      |
| dataset       | prevalencia                                        | 0.001214   | OK       | prevalencia real de celdas-día candidatas                                            |
| dataset       | celdas                                             | 399878     | OK       |                                                                                      |
| particiones   | train_dataset.parquet_filas                        | 6436147    | OK       |                                                                                      |
| particiones   | train_dataset.parquet_prevalencia                  | 0.093727   | OK       | case-control esperada ~9%                                                            |
| priors        | train_dataset.parquet_Chaco_pos_dist_hist_0_pct    | 53.5       | OK       | 100% indicaría fuga del propio evento                                                |
| priors        | train_dataset.parquet_Oriental_pos_dist_hist_0_pct | 69.96      | OK       | 100% indicaría fuga del propio evento                                                |
| particiones   | val_dataset.parquet_filas                          | 1016840    | OK       |                                                                                      |
| particiones   | val_dataset.parquet_prevalencia                    | 0.090909   | OK       | case-control esperada ~9%                                                            |
| priors        | val_dataset.parquet_Chaco_pos_dist_hist_0_pct      | 66.08      | OK       | 100% indicaría fuga del propio evento                                                |
| priors        | val_dataset.parquet_Oriental_pos_dist_hist_0_pct   | 85.67      | OK       | 100% indicaría fuga del propio evento                                                |
| particiones   | test_dataset.parquet_filas                         | 1841147    | OK       |                                                                                      |
| particiones   | test_dataset.parquet_prevalencia                   | 0.090917   | OK       | case-control esperada ~9%                                                            |
| priors        | test_dataset.parquet_Chaco_pos_dist_hist_0_pct     | 63.18      | OK       | 100% indicaría fuga del propio evento                                                |
| priors        | test_dataset.parquet_Oriental_pos_dist_hist_0_pct  | 78.73      | OK       | 100% indicaría fuga del propio evento                                                |
| fuga_espacial | pct_celdas_test_presentes_en_train                 | 99.97      | WARN     | el test es temporal; un holdout espacial requiere bloques/regiones                   |
| modelos       | lightgbm_hibrido_chaco.booster                     | presente   | OK       |                                                                                      |
| modelos       | lightgbm_hibrido_oriental.booster                  | presente   | OK       |                                                                                      |
| modelos       | lightgbm_propagacion_chaco.booster                 | presente   | OK       |                                                                                      |
| modelos       | lightgbm_propagacion_oriental.booster              | presente   | OK       |                                                                                      |
| manifiesto    | hashes_pendientes                                  | 3          | WARN     |                                                                                      |
| manifiesto    | filas                                              | 17         | OK       |                                                                                      |
