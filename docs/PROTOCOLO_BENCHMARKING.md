# Protocolo de Benchmarking Internacional para Validación de Modelos Predictivos de Incendios Forestales
## Especificación Metodológica y Diseño de Evaluación Comparativa

Documento técnico formal de benchmarking para el repositorio `Incendios_PY`.
Ver copia sincronizada en el vault de Obsidian del usuario en:
`/Users/martinsanchez/Documents/Martin/Nota captura/Protocolo de Benchmarking Internacional - Modelos de Incendios Forestales.md`

### Modelos Contendientes Evaluados:
1. **Incendios_PY (LightGBM + Platt Monótono + Two-Stage Gating):** Modelo GeoAI adaptado a Paraguay con 22 variables biofísicas y resolución de 1 km².
2. **Canadian FWI Puro (Van Wagner, 1987 / Copernicus EFFIS):** Estimador estándar internacional basado en balance hídrico y propagación inicial (ISI).
3. **Proxy INPE Risco de Fogo (Setzer et al., 2019):** Modelo exponencial semi-empírico de desecación por sequía consecutiva utilizado en Sudamérica.
4. **Línea Base Climatológica:** Frecuencia marginal histórica empírica regional.
