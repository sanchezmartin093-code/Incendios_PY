# Tabla 3. Ranking de Importancia de Variables por Ecorregión (TreeSHAP)

| region   | feature                      | nombre_legible                                   |   shap_importancia_media |   ranking |
|:---------|:-----------------------------|:-------------------------------------------------|-------------------------:|----------:|
| Chaco    | dist_foco_hist_km            | Distance to historical fire (km)                 |              0.486571    |         1 |
| Chaco    | pct_bosque_seco              | Dry Chaco forest (%)                             |              0.141371    |         2 |
| Chaco    | focalizacion_antropica_local | Local focalization index (5 km / 25 km)          |              0.0728805   |         3 |
| Chaco    | presion_desmonte_vpd         | Deforestation × VPD pressure                     |              0.0653986   |         4 |
| Chaco    | dist_corredor_antropico_km   | Distance to active anthropogenic corridor (km)   |              0.062795    |         5 |
| Chaco    | borde_bosque_agro_pct        | Forest–agriculture edge fraction (%)             |              0.0209901   |         6 |
| Chaco    | pct_desmonte_t1              | Tree-cover loss, Y-1 (%)                         |              0.020295    |         7 |
| Chaco    | densidad_hist_5km            | Local anthropogenic ignition density (5 km)      |              0.0201275   |         8 |
| Chaco    | doy_cos                      | doy_cos                                          |              0.0169211   |         9 |
| Chaco    | precipitacion_mm             | Daily rainfall (mm)                              |              0.0115138   |        10 |
| Chaco    | potencial_viento_cuadratico  | Dynamic wind pressure (v²)                       |              0.00752272  |        11 |
| Chaco    | presion_quema_pastura        | Pasture burning pressure (pasture × VPD × month) |              0.00596333  |        12 |
| Chaco    | swvl1_m3m3                   | Topsoil moisture 0–7 cm (m³/m³)                  |              0.00592673  |        13 |
| Chaco    | riesgo_estero_seco           | Dry estero risk (wetland × deficit)              |              0.00573363  |        14 |
| Chaco    | dia_del_anio                 | Day of year                                      |              0.00409044  |        15 |
| Chaco    | indice_ecotono               | Forest–pasture ecotone fragmentation index       |              0.00317482  |        16 |
| Chaco    | pct_pastura_ganadera         | Pasture cover (%)                                |              0.00292778  |        17 |
| Chaco    | isi                          | Initial spread index (ISI)                       |              0.00276592  |        18 |
| Chaco    | densidad_hist_25km           | Historical ignition density (25 km)              |              0.00203475  |        19 |
| Chaco    | interaccion_acceso_vpd       | Accessibility–desiccation synergy                |              0.00127704  |        20 |
| Chaco    | temp_min_c                   | Minimum temperature (°C)                         |              0.00126774  |        21 |
| Chaco    | mes                          | Month                                            |              0.00109998  |        22 |
| Chaco    | ffmc                         | Fine fuel moisture code (FFMC)                   |              0.00106255  |        23 |
| Chaco    | temporada_quema_ganadera     | temporada_quema_ganadera                         |              0.000875833 |        24 |
| Chaco    | deficit_hidrico_acumulado    | Accumulated water deficit (VPD × CDD)            |              0.000853064 |        25 |
| Chaco    | pct_sabana_pastizal          | Savanna / natural grassland cover (%)            |              0.000774658 |        26 |
| Chaco    | interaccion_viento_vpd       | Wind × VPD interaction                           |              0.000717013 |        27 |
| Chaco    | pct_agricultura_anual        | Annual agriculture cover (%)                     |              0.000607873 |        28 |
| Chaco    | pct_desmonte_t2              | Tree-cover loss, Y-2 (%)                         |              0.000571211 |        29 |
| Chaco    | deficit_hidro_60d            | Antecedent water deficit, 60 days (mm)           |              0.000560152 |        30 |
| Chaco    | doy_sin                      | doy_sin                                          |              0.000551661 |        31 |
| Chaco    | temp_max_c                   | Maximum temperature (°C)                         |              0.000293876 |        32 |
| Chaco    | dias_secos_consecutivos      | Consecutive dry days (CDD)                       |              0.000176997 |        33 |
| Chaco    | viento_max_kmh               | Maximum wind speed (km/h)                        |              0.000158218 |        34 |
| Chaco    | dia_semana                   | Day of week                                      |              7.13408e-05 |        35 |
| Chaco    | rango_termico_diario         | Daily thermal range (°C)                         |              4.27253e-05 |        36 |
| Chaco    | precipitacion_48h_mm         | 48-h accumulated rainfall (mm)                   |              3.61337e-05 |        37 |
| Chaco    | combustible_curado_helada    | Frost-cured pasture/grass fuel                   |              0           |        38 |
| Chaco    | fdd_heladas_60d              | Antecedent frost severity, 60 days (°C-days)     |              0           |        39 |
| Chaco    | es_fin_de_semana             | Weekend effect (anthropogenic)                   |              0           |        40 |
| Chaco    | pct_plantacion_forestal      | Forest plantation cover (%)                      |              0           |        41 |
| Chaco    | densidad_hist_10km           | Local anthropogenic ignition density (10 km)     |              0           |        42 |
| Chaco    | pct_humedal_palmar           | Flooded grassland / estero cover (%)             |              0           |        43 |
| Chaco    | vpd_max_kpa                  | Vapor pressure deficit (VPD, kPa)                |              0           |        44 |
| Oriental | dist_foco_hist_km            | Distance to historical fire (km)                 |              0.198766    |         1 |
| Oriental | focalizacion_antropica_local | Local focalization index (5 km / 25 km)          |              0.194319    |         2 |
| Oriental | pct_agricultura_anual        | Annual agriculture cover (%)                     |              0.169538    |         3 |
| Oriental | dist_corredor_antropico_km   | Distance to active anthropogenic corridor (km)   |              0.098708    |         4 |
| Oriental | presion_desmonte_vpd         | Deforestation × VPD pressure                     |              0.0986837   |         5 |
| Oriental | pct_humedal_palmar           | Flooded grassland / estero cover (%)             |              0.0789221   |         6 |
| Oriental | freq_hist_1km                | Chronic recurrence in a 1-km² cell               |              0.0737129   |         7 |
| Oriental | pct_sabana_pastizal          | Savanna / natural grassland cover (%)            |              0.0473967   |         8 |
| Oriental | riesgo_estero_seco           | Dry estero risk (wetland × deficit)              |              0.0444481   |         9 |
| Oriental | pct_desmonte_t1              | Tree-cover loss, Y-1 (%)                         |              0.0350678   |        10 |
| Oriental | pct_desmonte_t2              | Tree-cover loss, Y-2 (%)                         |              0.0307428   |        11 |
| Oriental | borde_bosque_agro_pct        | Forest–agriculture edge fraction (%)             |              0.0181525   |        12 |
| Oriental | doy_cos                      | doy_cos                                          |              0.0175653   |        13 |
| Oriental | densidad_hist_5km            | Local anthropogenic ignition density (5 km)      |              0.0151603   |        14 |
| Oriental | precipitacion_mm             | Daily rainfall (mm)                              |              0.0137516   |        15 |
| Oriental | ffmc                         | Fine fuel moisture code (FFMC)                   |              0.0125672   |        16 |
| Oriental | isi                          | Initial spread index (ISI)                       |              0.0121584   |        17 |
| Oriental | precipitacion_48h_mm         | 48-h accumulated rainfall (mm)                   |              0.0109464   |        18 |
| Oriental | deficit_hidrico_acumulado    | Accumulated water deficit (VPD × CDD)            |              0.0107567   |        19 |
| Oriental | dia_del_anio                 | Day of year                                      |              0.0105194   |        20 |
| Oriental | dias_secos_consecutivos      | Consecutive dry days (CDD)                       |              0.00721487  |        21 |
| Oriental | deficit_hidro_60d            | Antecedent water deficit, 60 days (mm)           |              0.00631234  |        22 |
| Oriental | presion_quema_pastura        | Pasture burning pressure (pasture × VPD × month) |              0.00611145  |        23 |
| Oriental | pct_pastura_ganadera         | Pasture cover (%)                                |              0.00599626  |        24 |
| Oriental | indice_ecotono               | Forest–pasture ecotone fragmentation index       |              0.00486563  |        25 |
| Oriental | temp_min_c                   | Minimum temperature (°C)                         |              0.00287861  |        26 |
| Oriental | temp_max_c                   | Maximum temperature (°C)                         |              0.00286508  |        27 |
| Oriental | mes                          | Month                                            |              0.00243083  |        28 |
| Oriental | doy_sin                      | doy_sin                                          |              0.00190419  |        29 |
| Oriental | interaccion_viento_vpd       | Wind × VPD interaction                           |              0.00158902  |        30 |
| Oriental | rango_termico_diario         | Daily thermal range (°C)                         |              0.00134605  |        31 |
| Oriental | pct_bosque_humedo            | Atlantic forest BAAPA (%)                        |              0.00133642  |        32 |
| Oriental | densidad_hist_10km           | Local anthropogenic ignition density (10 km)     |              0.001089    |        33 |
| Oriental | interaccion_acceso_vpd       | Accessibility–desiccation synergy                |              0.000706009 |        34 |
| Oriental | potencial_viento_cuadratico  | Dynamic wind pressure (v²)                       |              0.000673135 |        35 |
| Oriental | vpd_max_kpa                  | Vapor pressure deficit (VPD, kPa)                |              0.000525717 |        36 |
| Oriental | swvl1_m3m3                   | Topsoil moisture 0–7 cm (m³/m³)                  |              0.000269832 |        37 |
| Oriental | densidad_hist_25km           | Historical ignition density (25 km)              |              0.000183168 |        38 |
| Oriental | viento_max_kmh               | Maximum wind speed (km/h)                        |              0.000170542 |        39 |
| Oriental | dia_semana                   | Day of week                                      |              0.000112542 |        40 |
| Oriental | pct_plantacion_forestal      | Forest plantation cover (%)                      |              6.0485e-05  |        41 |
| Oriental | combustible_curado_helada    | Frost-cured pasture/grass fuel                   |              2.41227e-05 |        42 |
| Oriental | fdd_heladas_60d              | Antecedent frost severity, 60 days (°C-days)     |              1.08877e-05 |        43 |
| Oriental | temporada_quema_ganadera     | temporada_quema_ganadera                         |              0           |        44 |
| Oriental | es_fin_de_semana             | Weekend effect (anthropogenic)                   |              0           |        45 |
