# Validación de integridad de datos

| seccion           | chequeo                                     | valor      | estado   | nota                                                                                 |
|:------------------|:--------------------------------------------|:-----------|:---------|:-------------------------------------------------------------------------------------|
| grilla            | filas                                       | 400006     | OK       |                                                                                      |
| grilla            | cell_id_unicos                              | 400006     | OK       |                                                                                      |
| grilla            | celdas_Chaco                                | 240880     | OK       |                                                                                      |
| grilla            | celdas_Oriental                             | 159126     | OK       |                                                                                      |
| clima             | filas                                       | 1870080    | OK       |                                                                                      |
| clima             | nodos                                       | 640        | OK       |                                                                                      |
| clima             | fecha_min                                   | 2018-01-01 | OK       |                                                                                      |
| clima             | fecha_max                                   | 2025-12-31 | OK       |                                                                                      |
| clima             | precipitacion_mayor_300mm                   | 0          | OK       | valores posibles pero improbables; revisar desacumulación                            |
| clima             | precipitacion_max_mm                        | 237.383    | OK       | pendiente ejecutar scripts/02c_corregir_clima_openmeteo.py cuando haya cuota externa |
| clima             | et0_distinto_de_cero                        | 1870080    | OK       | 0 indica que 02f/02c no se ejecutó                                                   |
| clima             | rafaga_es_proxy_1.3x                        | 4e-06      | WARN     |                                                                                      |
| clima             | swvl1_m3m3_nulos                            | 0          | OK       |                                                                                      |
| clima             | swvl1_m3m3_min                              | 0.0677     | OK       |                                                                                      |
| clima             | swvl1_m3m3_max                              | 0.5117     | OK       |                                                                                      |
| clima             | fdd_heladas_60d_nulos                       | 0          | OK       |                                                                                      |
| clima             | fdd_heladas_60d_min                         | 0.0        | OK       |                                                                                      |
| clima             | fdd_heladas_60d_max                         | 6.539      | OK       |                                                                                      |
| clima             | deficit_hidro_60d_nulos                     | 0          | OK       |                                                                                      |
| clima             | deficit_hidro_60d_min                       | -704.49    | OK       |                                                                                      |
| clima             | deficit_hidro_60d_max                       | 394.101    | OK       |                                                                                      |
| combustible       | pct_pastura_ganadera_presente               | 1          | OK       |                                                                                      |
| combustible       | pct_agricultura_anual_presente              | 1          | OK       |                                                                                      |
| combustible       | pct_plantacion_forestal_presente            | 1          | OK       |                                                                                      |
| combustible       | borde_bosque_agro_pct_presente              | 1          | OK       |                                                                                      |
| combustible       | suma_componentes_max_error                  | 0.0        | OK       |                                                                                      |
| combustible       | borde_max                                   | 84.888     | OK       |                                                                                      |
| combustible_anual | filas                                       | 3600054    | OK       |                                                                                      |
| combustible_anual | anios                                       | 2018-2026  | OK       |                                                                                      |
| combustible_anual | fuel_lag_min                                | 1          | OK       |                                                                                      |
| combustible_anual | filas_mapa_no_previo                        | 0          | OK       |                                                                                      |
| combustible_anual | suma_componentes_max_error                  | 0.0        | OK       |                                                                                      |
| desmonte          | filas                                       | 3600054    | OK       |                                                                                      |
| desmonte          | anios                                       | 2018-2026  | OK       |                                                                                      |
| desmonte          | pct_t1_max                                  | 100.0      | OK       |                                                                                      |
| desmonte          | nulos_t1                                    | 0          | OK       |                                                                                      |
| firms             | celda_dias_positivos                        | 892533     | OK       |                                                                                      |
| firms             | fecha_min                                   | 2018-01-01 | OK       |                                                                                      |
| firms             | fecha_max                                   | 2026-01-04 | WARN     | puede exceder la ventana de clima                                                    |
| firms_raw         | detecciones                                 | 3685836    | OK       |                                                                                      |
| firms_raw         | scan_max_km                                 | 0.8        | WARN     | si max_scan<=1.5 el filtro Bow-Tie no descarta registros                             |
| firms_raw         | detecciones_type_no_0                       | 5319       | WARN     | no se filtró type; revisar fuentes industriales/offshore                             |
| dataset           | filas                                       | 915561114  | OK       |                                                                                      |
| dataset           | fuegos                                      | 867430     | OK       |                                                                                      |
| dataset           | prevalencia                                 | 0.000947   | OK       | prevalencia real de celdas-día candidatas                                            |
| dataset           | celdas                                      | 397448     | OK       |                                                                                      |
| particiones       | train_ignicion_features_v3_faltantes        | 0          | OK       | contrato v3 completo                                                                 |
| particiones       | train_ignicion_filas                        | 1018787    | OK       |                                                                                      |
| particiones       | train_ignicion_prevalencia                  | 0.090909   | OK       | case-control esperada ~9%                                                            |
| particiones       | train_ignicion_fuel_futuro                  | 0          | OK       |                                                                                      |
| particiones       | train_ignicion_fuel_lag_min                 | 1          | OK       |                                                                                      |
| priors            | train_ignicion_Chaco_pos_dist_hist_0_pct    | 47.46      | OK       | 100% indicaría fuga del propio evento                                                |
| priors            | train_ignicion_Oriental_pos_dist_hist_0_pct | 62.01      | OK       | 100% indicaría fuga del propio evento                                                |
| particiones       | val_ignicion_features_v3_faltantes          | 0          | OK       | contrato v3 completo                                                                 |
| particiones       | val_ignicion_filas                          | 188507     | OK       |                                                                                      |
| particiones       | val_ignicion_prevalencia                    | 0.090909   | OK       | case-control esperada ~9%                                                            |
| particiones       | val_ignicion_fuel_futuro                    | 0          | OK       |                                                                                      |
| particiones       | val_ignicion_fuel_lag_min                   | 1          | OK       |                                                                                      |
| priors            | val_ignicion_Chaco_pos_dist_hist_0_pct      | 59.75      | OK       | 100% indicaría fuga del propio evento                                                |
| priors            | val_ignicion_Oriental_pos_dist_hist_0_pct   | 78.85      | OK       | 100% indicaría fuga del propio evento                                                |
| particiones       | test_ignicion_features_v3_faltantes         | 0          | OK       | contrato v3 completo                                                                 |
| particiones       | test_ignicion_filas                         | 376552     | OK       |                                                                                      |
| particiones       | test_ignicion_prevalencia                   | 0.090909   | OK       | case-control esperada ~9%                                                            |
| particiones       | test_ignicion_fuel_futuro                   | 0          | OK       |                                                                                      |
| particiones       | test_ignicion_fuel_lag_min                  | 1          | OK       |                                                                                      |
| priors            | test_ignicion_Chaco_pos_dist_hist_0_pct     | 58.78      | OK       | 100% indicaría fuga del propio evento                                                |
| priors            | test_ignicion_Oriental_pos_dist_hist_0_pct  | 71.92      | OK       | 100% indicaría fuga del propio evento                                                |
| fuga_espacial     | pct_celdas_test_presentes_en_train          | 90.97      | WARN     | el test es temporal; un holdout espacial requiere bloques/regiones                   |
| modelos           | lightgbm_hibrido_chaco.booster              | presente   | OK       |                                                                                      |
| modelos           | lightgbm_hibrido_oriental.booster           | presente   | OK       |                                                                                      |
| modelos           | lightgbm_propagacion_chaco.booster          | presente   | OK       |                                                                                      |
| modelos           | lightgbm_propagacion_oriental.booster       | presente   | OK       |                                                                                      |
| manifiesto        | hashes_pendientes                           | 0          | OK       |                                                                                      |
| manifiesto        | filas                                       | 44         | OK       |                                                                                      |
