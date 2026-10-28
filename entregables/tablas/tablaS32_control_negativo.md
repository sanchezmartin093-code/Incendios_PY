# Table S32. Negative control and permutation importance

| Region | Setting | Negative-control AUC |
|---|---|---:|
| Chaco | all features | 0.6297 |
| Chaco | without historical | 0.6276 |
| Oriental | all features | 0.5138 |
| Oriental | without historical | 0.5168 |

| Region | Family | Features | ΔROC |
|---|---|---:|---:|
| Chaco | weather | 18 | 0.0034 |
| Chaco | fuel | 11 | 0.0282 |
| Chaco | anthropogenic | 7 | 0.0234 |
| Chaco | historical | 2 | 0.1701 |
| Chaco | calendar | 6 | 0.0018 |
| Oriental | weather | 18 | 0.0128 |
| Oriental | fuel | 11 | 0.0411 |
| Oriental | anthropogenic | 7 | 0.0408 |
| Oriental | historical | 3 | 0.0342 |
| Oriental | calendar | 6 | 0.0022 |

| Region | Feature | ΔROC |
|---|---|---:|
| Chaco | `dist_foco_hist_km` | 0.1698 |
| Chaco | `presion_desmonte_vpd` | 0.0096 |
| Chaco | `pct_bosque_seco` | 0.0079 |
| Chaco | `dist_corredor_antropico_km` | 0.0078 |
| Chaco | `focalizacion_antropica_local` | 0.0038 |
| Chaco | `pct_desmonte_t1` | 0.0026 |
| Chaco | `doy_cos` | 0.0016 |
| Chaco | `precipitacion_mm` | 0.0010 |
| Chaco | `riesgo_estero_seco` | 0.0007 |
| Chaco | `swvl1_m3m3` | 0.0007 |
| Oriental | `dist_foco_hist_km` | 0.0200 |
| Oriental | `focalizacion_antropica_local` | 0.0164 |
| Oriental | `dist_corredor_antropico_km` | 0.0060 |
| Oriental | `pct_agricultura_anual` | 0.0059 |
| Oriental | `freq_hist_1km` | 0.0048 |
| Oriental | `pct_humedal_palmar` | 0.0042 |
| Oriental | `riesgo_estero_seco` | 0.0038 |
| Oriental | `presion_desmonte_vpd` | 0.0032 |
| Oriental | `pct_sabana_pastizal` | 0.0016 |
| Oriental | `ffmc` | 0.0013 |
