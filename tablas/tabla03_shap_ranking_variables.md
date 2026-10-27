# Tabla 3. Ranking de Importancia de Variables por Ecorregión (TreeSHAP)

| region   | feature                      | nombre_legible                                   |   shap_importancia_media |   ranking |
|:---------|:-----------------------------|:-------------------------------------------------|-------------------------:|----------:|
| Chaco    | dist_foco_hist_km            | Distance to historical fire (km)                 |              0.488547    |         1 |
| Chaco    | pct_bosque_seco              | Dry Chaco forest (%)                             |              0.138441    |         2 |
| Chaco    | focalizacion_antropica_local | Local focalization index (5 km / 25 km)          |              0.0776825   |         3 |
| Chaco    | dist_corredor_antropico_km   | Distance to active anthropogenic corridor (km)   |              0.0627689   |         4 |
| Chaco    | presion_desmonte_vpd         | Deforestation × VPD pressure                     |              0.0620125   |         5 |
| Chaco    | densidad_hist_5km            | Local anthropogenic ignition density (5 km)      |              0.0219523   |         6 |
| Chaco    | pct_desmonte_t1              | Tree-cover loss, Y-1 (%)                         |              0.021775    |         7 |
| Chaco    | doy_cos                      | doy_cos                                          |              0.0184024   |         8 |
| Chaco    | borde_bosque_agro_pct        | Forest–agriculture edge fraction (%)             |              0.0145725   |         9 |
| Chaco    | precipitacion_mm             | Daily rainfall (mm)                              |              0.0116078   |        10 |
| Chaco    | potencial_viento_cuadratico  | Dynamic wind pressure (v²)                       |              0.00921907  |        11 |
| Chaco    | presion_quema_pastura        | Pasture burning pressure (pasture × VPD × month) |              0.00643015  |        12 |
| Chaco    | swvl1_m3m3                   | Topsoil moisture 0–7 cm (m³/m³)                  |              0.00538095  |        13 |
| Chaco    | riesgo_estero_seco           | Dry estero risk (wetland × deficit)              |              0.00464681  |        14 |
| Chaco    | pct_pastura_ganadera         | Pasture cover (%)                                |              0.00443529  |        15 |
| Chaco    | indice_ecotono               | Forest–pasture ecotone fragmentation index       |              0.00408785  |        16 |
| Chaco    | densidad_hist_25km           | Historical ignition density (25 km)              |              0.0032915   |        17 |
| Chaco    | isi                          | Initial spread index (ISI)                       |              0.00278949  |        18 |
| Chaco    | dia_del_anio                 | Day of year                                      |              0.00212971  |        19 |
| Chaco    | temporada_quema_ganadera     | temporada_quema_ganadera                         |              0.00151529  |        20 |
| Chaco    | ffmc                         | Fine fuel moisture code (FFMC)                   |              0.00130926  |        21 |
| Chaco    | temp_min_c                   | Minimum temperature (°C)                         |              0.00125604  |        22 |
| Chaco    | pct_sabana_pastizal          | Savanna / natural grassland cover (%)            |              0.00123643  |        23 |
| Chaco    | deficit_hidro_60d            | Antecedent water deficit, 60 days (mm)           |              0.000777346 |        24 |
| Chaco    | mes                          | Month                                            |              0.000728911 |        25 |
| Chaco    | deficit_hidrico_acumulado    | Accumulated water deficit (VPD × CDD)            |              0.000728317 |        26 |
| Chaco    | interaccion_viento_vpd       | Wind × VPD interaction                           |              0.000704211 |        27 |
| Chaco    | pct_agricultura_anual        | Annual agriculture cover (%)                     |              0.000674236 |        28 |
| Chaco    | doy_sin                      | doy_sin                                          |              0.00067077  |        29 |
| Chaco    | interaccion_acceso_vpd       | Accessibility–desiccation synergy                |              0.000445918 |        30 |
| Chaco    | rango_termico_diario         | Daily thermal range (°C)                         |              0.000413034 |        31 |
| Chaco    | temp_max_c                   | Maximum temperature (°C)                         |              0.000341818 |        32 |
| Chaco    | dia_semana                   | Day of week                                      |              0.000224876 |        33 |
| Chaco    | dias_secos_consecutivos      | Consecutive dry days (CDD)                       |              0.000193906 |        34 |
| Chaco    | densidad_hist_10km           | Local anthropogenic ignition density (10 km)     |              0.000192146 |        35 |
| Chaco    | pct_desmonte_t2              | Tree-cover loss, Y-2 (%)                         |              0.000157047 |        36 |
| Chaco    | viento_max_kmh               | Maximum wind speed (km/h)                        |              5.90499e-05 |        37 |
| Chaco    | pct_humedal_palmar           | Flooded grassland / estero cover (%)             |              0           |        38 |
| Chaco    | pct_plantacion_forestal      | Forest plantation cover (%)                      |              0           |        39 |
| Chaco    | es_fin_de_semana             | Weekend effect (anthropogenic)                   |              0           |        40 |
| Chaco    | combustible_curado_helada    | Frost-cured pasture/grass fuel                   |              0           |        41 |
| Chaco    | fdd_heladas_60d              | Antecedent frost severity, 60 days (°C-days)     |              0           |        42 |
| Chaco    | precipitacion_48h_mm         | 48-h accumulated rainfall (mm)                   |              0           |        43 |
| Chaco    | vpd_max_kpa                  | Vapor pressure deficit (VPD, kPa)                |              0           |        44 |
| Oriental | dist_foco_hist_km            | Distance to historical fire (km)                 |              0.222959    |         1 |
| Oriental | focalizacion_antropica_local | Local focalization index (5 km / 25 km)          |              0.201239    |         2 |
| Oriental | pct_agricultura_anual        | Annual agriculture cover (%)                     |              0.166784    |         3 |
| Oriental | dist_corredor_antropico_km   | Distance to active anthropogenic corridor (km)   |              0.101604    |         4 |
| Oriental | presion_desmonte_vpd         | Deforestation × VPD pressure                     |              0.0909603   |         5 |
| Oriental | pct_humedal_palmar           | Flooded grassland / estero cover (%)             |              0.0701217   |         6 |
| Oriental | freq_hist_1km                | Chronic recurrence in a 1-km² cell               |              0.0567534   |         7 |
| Oriental | riesgo_estero_seco           | Dry estero risk (wetland × deficit)              |              0.0486838   |         8 |
| Oriental | pct_sabana_pastizal          | Savanna / natural grassland cover (%)            |              0.0438298   |         9 |
| Oriental | pct_desmonte_t1              | Tree-cover loss, Y-1 (%)                         |              0.0333243   |        10 |
| Oriental | pct_desmonte_t2              | Tree-cover loss, Y-2 (%)                         |              0.0331574   |        11 |
| Oriental | borde_bosque_agro_pct        | Forest–agriculture edge fraction (%)             |              0.0177089   |        12 |
| Oriental | precipitacion_mm             | Daily rainfall (mm)                              |              0.0173002   |        13 |
| Oriental | doy_cos                      | doy_cos                                          |              0.0144009   |        14 |
| Oriental | densidad_hist_5km            | Local anthropogenic ignition density (5 km)      |              0.013985    |        15 |
| Oriental | ffmc                         | Fine fuel moisture code (FFMC)                   |              0.0131939   |        16 |
| Oriental | deficit_hidrico_acumulado    | Accumulated water deficit (VPD × CDD)            |              0.0121707   |        17 |
| Oriental | precipitacion_48h_mm         | 48-h accumulated rainfall (mm)                   |              0.0105795   |        18 |
| Oriental | dia_del_anio                 | Day of year                                      |              0.0102407   |        19 |
| Oriental | isi                          | Initial spread index (ISI)                       |              0.00786703  |        20 |
| Oriental | indice_ecotono               | Forest–pasture ecotone fragmentation index       |              0.00702473  |        21 |
| Oriental | dias_secos_consecutivos      | Consecutive dry days (CDD)                       |              0.0059823   |        22 |
| Oriental | presion_quema_pastura        | Pasture burning pressure (pasture × VPD × month) |              0.00581707  |        23 |
| Oriental | deficit_hidro_60d            | Antecedent water deficit, 60 days (mm)           |              0.00476468  |        24 |
| Oriental | pct_pastura_ganadera         | Pasture cover (%)                                |              0.0037446   |        25 |
| Oriental | doy_sin                      | doy_sin                                          |              0.00235103  |        26 |
| Oriental | temp_min_c                   | Minimum temperature (°C)                         |              0.00213301  |        27 |
| Oriental | densidad_hist_10km           | Local anthropogenic ignition density (10 km)     |              0.00195855  |        28 |
| Oriental | mes                          | Month                                            |              0.00190296  |        29 |
| Oriental | interaccion_acceso_vpd       | Accessibility–desiccation synergy                |              0.00179748  |        30 |
| Oriental | rango_termico_diario         | Daily thermal range (°C)                         |              0.00161874  |        31 |
| Oriental | temp_max_c                   | Maximum temperature (°C)                         |              0.00158103  |        32 |
| Oriental | potencial_viento_cuadratico  | Dynamic wind pressure (v²)                       |              0.00154475  |        33 |
| Oriental | interaccion_viento_vpd       | Wind × VPD interaction                           |              0.00104554  |        34 |
| Oriental | pct_bosque_humedo            | Atlantic forest BAAPA (%)                        |              0.00100498  |        35 |
| Oriental | vpd_max_kpa                  | Vapor pressure deficit (VPD, kPa)                |              0.000640842 |        36 |
| Oriental | swvl1_m3m3                   | Topsoil moisture 0–7 cm (m³/m³)                  |              0.000491167 |        37 |
| Oriental | viento_max_kmh               | Maximum wind speed (km/h)                        |              0.000223413 |        38 |
| Oriental | dia_semana                   | Day of week                                      |              0.000107847 |        39 |
| Oriental | densidad_hist_25km           | Historical ignition density (25 km)              |              8.48026e-05 |        40 |
| Oriental | pct_plantacion_forestal      | Forest plantation cover (%)                      |              3.96346e-05 |        41 |
| Oriental | fdd_heladas_60d              | Antecedent frost severity, 60 days (°C-days)     |              2.05285e-06 |        42 |
| Oriental | temporada_quema_ganadera     | temporada_quema_ganadera                         |              0           |        43 |
| Oriental | combustible_curado_helada    | Frost-cured pasture/grass fuel                   |              0           |        44 |
| Oriental | es_fin_de_semana             | Weekend effect (anthropogenic)                   |              0           |        45 |
