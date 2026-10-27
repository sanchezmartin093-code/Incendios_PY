# Tabla 3. Ranking de Importancia de Variables por Ecorregión (TreeSHAP)

| region   | feature                      | nombre_legible                                   |   shap_importancia_media |   ranking |
|:---------|:-----------------------------|:-------------------------------------------------|-------------------------:|----------:|
| Chaco    | dist_foco_hist_km            | Distance to historical fire (km)                 |              0.486099    |         1 |
| Chaco    | pct_bosque_seco              | Dry Chaco forest (%)                             |              0.141103    |         2 |
| Chaco    | focalizacion_antropica_local | Local focalization index (5 km / 25 km)          |              0.0733745   |         3 |
| Chaco    | presion_desmonte_vpd         | Deforestation × VPD pressure                     |              0.0658379   |         4 |
| Chaco    | dist_corredor_antropico_km   | Distance to active anthropogenic corridor (km)   |              0.0624675   |         5 |
| Chaco    | densidad_hist_5km            | Local anthropogenic ignition density (5 km)      |              0.0198531   |         6 |
| Chaco    | pct_desmonte_t1              | Tree-cover loss, Y-1 (%)                         |              0.0197777   |         7 |
| Chaco    | doy_cos                      | doy_cos                                          |              0.0174362   |         8 |
| Chaco    | borde_bosque_agro_pct        | Forest–agriculture edge fraction (%)             |              0.0172089   |         9 |
| Chaco    | precipitacion_mm             | Daily rainfall (mm)                              |              0.0103223   |        10 |
| Chaco    | potencial_viento_cuadratico  | Dynamic wind pressure (v²)                       |              0.007349    |        11 |
| Chaco    | presion_quema_pastura        | Pasture burning pressure (pasture × VPD × month) |              0.00688531  |        12 |
| Chaco    | swvl1_m3m3                   | Topsoil moisture 0–7 cm (m³/m³)                  |              0.00551266  |        13 |
| Chaco    | riesgo_estero_seco           | Dry estero risk (wetland × deficit)              |              0.00496748  |        14 |
| Chaco    | indice_ecotono               | Forest–pasture ecotone fragmentation index       |              0.00418968  |        15 |
| Chaco    | isi                          | Initial spread index (ISI)                       |              0.00335546  |        16 |
| Chaco    | pct_pastura_ganadera         | Pasture cover (%)                                |              0.00334852  |        17 |
| Chaco    | densidad_hist_25km           | Historical ignition density (25 km)              |              0.00295332  |        18 |
| Chaco    | dia_del_anio                 | Day of year                                      |              0.00247916  |        19 |
| Chaco    | temp_min_c                   | Minimum temperature (°C)                         |              0.00139346  |        20 |
| Chaco    | doy_sin                      | doy_sin                                          |              0.00124421  |        21 |
| Chaco    | pct_sabana_pastizal          | Savanna / natural grassland cover (%)            |              0.0012112   |        22 |
| Chaco    | mes                          | Month                                            |              0.00107196  |        23 |
| Chaco    | ffmc                         | Fine fuel moisture code (FFMC)                   |              0.00106654  |        24 |
| Chaco    | deficit_hidrico_acumulado    | Accumulated water deficit (VPD × CDD)            |              0.000880651 |        25 |
| Chaco    | interaccion_viento_vpd       | Wind × VPD interaction                           |              0.000691459 |        26 |
| Chaco    | temporada_quema_ganadera     | temporada_quema_ganadera                         |              0.000641743 |        27 |
| Chaco    | pct_desmonte_t2              | Tree-cover loss, Y-2 (%)                         |              0.000454194 |        28 |
| Chaco    | interaccion_acceso_vpd       | Accessibility–desiccation synergy                |              0.000410017 |        29 |
| Chaco    | pct_agricultura_anual        | Annual agriculture cover (%)                     |              0.000405526 |        30 |
| Chaco    | deficit_hidro_60d            | Antecedent water deficit, 60 days (mm)           |              0.000197732 |        31 |
| Chaco    | viento_max_kmh               | Maximum wind speed (km/h)                        |              0.000170772 |        32 |
| Chaco    | temp_max_c                   | Maximum temperature (°C)                         |              0.000146127 |        33 |
| Chaco    | dia_semana                   | Day of week                                      |              8.50081e-05 |        34 |
| Chaco    | dias_secos_consecutivos      | Consecutive dry days (CDD)                       |              4.94105e-05 |        35 |
| Chaco    | rango_termico_diario         | Daily thermal range (°C)                         |              4.27001e-05 |        36 |
| Chaco    | pct_humedal_palmar           | Flooded grassland / estero cover (%)             |              0           |        37 |
| Chaco    | pct_plantacion_forestal      | Forest plantation cover (%)                      |              0           |        38 |
| Chaco    | es_fin_de_semana             | Weekend effect (anthropogenic)                   |              0           |        39 |
| Chaco    | combustible_curado_helada    | Frost-cured pasture/grass fuel                   |              0           |        40 |
| Chaco    | fdd_heladas_60d              | Antecedent frost severity, 60 days (°C-days)     |              0           |        41 |
| Chaco    | densidad_hist_10km           | Local anthropogenic ignition density (10 km)     |              0           |        42 |
| Chaco    | precipitacion_48h_mm         | 48-h accumulated rainfall (mm)                   |              0           |        43 |
| Chaco    | vpd_max_kpa                  | Vapor pressure deficit (VPD, kPa)                |              0           |        44 |
| Oriental | dist_foco_hist_km            | Distance to historical fire (km)                 |              0.213364    |         1 |
| Oriental | focalizacion_antropica_local | Local focalization index (5 km / 25 km)          |              0.195768    |         2 |
| Oriental | pct_agricultura_anual        | Annual agriculture cover (%)                     |              0.172094    |         3 |
| Oriental | dist_corredor_antropico_km   | Distance to active anthropogenic corridor (km)   |              0.105301    |         4 |
| Oriental | presion_desmonte_vpd         | Deforestation × VPD pressure                     |              0.0934252   |         5 |
| Oriental | pct_humedal_palmar           | Flooded grassland / estero cover (%)             |              0.0745342   |         6 |
| Oriental | freq_hist_1km                | Chronic recurrence in a 1-km² cell               |              0.0581355   |         7 |
| Oriental | riesgo_estero_seco           | Dry estero risk (wetland × deficit)              |              0.0461181   |         8 |
| Oriental | pct_sabana_pastizal          | Savanna / natural grassland cover (%)            |              0.0454373   |         9 |
| Oriental | pct_desmonte_t1              | Tree-cover loss, Y-1 (%)                         |              0.0366119   |        10 |
| Oriental | pct_desmonte_t2              | Tree-cover loss, Y-2 (%)                         |              0.0313661   |        11 |
| Oriental | borde_bosque_agro_pct        | Forest–agriculture edge fraction (%)             |              0.0184475   |        12 |
| Oriental | precipitacion_mm             | Daily rainfall (mm)                              |              0.0179861   |        13 |
| Oriental | doy_cos                      | doy_cos                                          |              0.0158602   |        14 |
| Oriental | densidad_hist_5km            | Local anthropogenic ignition density (5 km)      |              0.0145699   |        15 |
| Oriental | ffmc                         | Fine fuel moisture code (FFMC)                   |              0.0129128   |        16 |
| Oriental | deficit_hidrico_acumulado    | Accumulated water deficit (VPD × CDD)            |              0.0111176   |        17 |
| Oriental | precipitacion_48h_mm         | 48-h accumulated rainfall (mm)                   |              0.0110329   |        18 |
| Oriental | dia_del_anio                 | Day of year                                      |              0.0104094   |        19 |
| Oriental | isi                          | Initial spread index (ISI)                       |              0.010355    |        20 |
| Oriental | dias_secos_consecutivos      | Consecutive dry days (CDD)                       |              0.00584071  |        21 |
| Oriental | presion_quema_pastura        | Pasture burning pressure (pasture × VPD × month) |              0.00518662  |        22 |
| Oriental | pct_pastura_ganadera         | Pasture cover (%)                                |              0.00503101  |        23 |
| Oriental | indice_ecotono               | Forest–pasture ecotone fragmentation index       |              0.00377824  |        24 |
| Oriental | deficit_hidro_60d            | Antecedent water deficit, 60 days (mm)           |              0.00351732  |        25 |
| Oriental | doy_sin                      | doy_sin                                          |              0.00232564  |        26 |
| Oriental | temp_max_c                   | Maximum temperature (°C)                         |              0.00232122  |        27 |
| Oriental | mes                          | Month                                            |              0.00198807  |        28 |
| Oriental | interaccion_viento_vpd       | Wind × VPD interaction                           |              0.00181589  |        29 |
| Oriental | interaccion_acceso_vpd       | Accessibility–desiccation synergy                |              0.00153674  |        30 |
| Oriental | temp_min_c                   | Minimum temperature (°C)                         |              0.001499    |        31 |
| Oriental | rango_termico_diario         | Daily thermal range (°C)                         |              0.00135085  |        32 |
| Oriental | pct_bosque_humedo            | Atlantic forest BAAPA (%)                        |              0.00120308  |        33 |
| Oriental | densidad_hist_10km           | Local anthropogenic ignition density (10 km)     |              0.0010601   |        34 |
| Oriental | potencial_viento_cuadratico  | Dynamic wind pressure (v²)                       |              0.00042103  |        35 |
| Oriental | densidad_hist_25km           | Historical ignition density (25 km)              |              0.000334406 |        36 |
| Oriental | swvl1_m3m3                   | Topsoil moisture 0–7 cm (m³/m³)                  |              0.00033121  |        37 |
| Oriental | vpd_max_kpa                  | Vapor pressure deficit (VPD, kPa)                |              0.000166614 |        38 |
| Oriental | pct_plantacion_forestal      | Forest plantation cover (%)                      |              0.000145558 |        39 |
| Oriental | viento_max_kmh               | Maximum wind speed (km/h)                        |              7.01366e-05 |        40 |
| Oriental | fdd_heladas_60d              | Antecedent frost severity, 60 days (°C-days)     |              0           |        41 |
| Oriental | combustible_curado_helada    | Frost-cured pasture/grass fuel                   |              0           |        42 |
| Oriental | temporada_quema_ganadera     | temporada_quema_ganadera                         |              0           |        43 |
| Oriental | dia_semana                   | Day of week                                      |              0           |        44 |
| Oriental | es_fin_de_semana             | Weekend effect (anthropogenic)                   |              0           |        45 |
