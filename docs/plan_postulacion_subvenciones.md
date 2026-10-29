# 📋 Plan Estratégico de Postulación a Subvenciones de Cómputo (Google for Startups & Google TRC)

> **Fecha de Actualización:** 28 de Octubre de 2026
> **Perfil del Postulante:** Investigador Independiente en Ciencia de Datos, GeoAI y Tecnología Cívica (Sin afiliación institucional formal / sin correo `.edu`).
> **Repositorio Insignia:** [https://github.com/sanchezmartin093-code/Incendios_PY](https://github.com/sanchezmartin093-code/Incendios_PY) (Release v2.1.0, Licencia MIT, 76 tests pasando).
> **Target de Publicación:** *Computers & Geosciences* (Elsevier).

---

## 🎯 1. Resumen de Oportunidades y Metas

Ambos programas son **100% compatibles y complementarios**, ya que pertenecen a ramas independientes de Google (Google Cloud vs. Google Research).

| Programa | Objetivo Operativo | Qué se obtiene | Probabilidad | Tiempo Respuesta |
| :--- | :--- | :--- | :---: | :---: |
| **Google for Startups Cloud (Nivel "Start")** | Despliegue de API, almacenamiento Parquet y **Tokens Vertex AI (Gemini 3.8 Flash Multimodal)** | **$2.000 USD** en créditos GCP (Vertex AI, Cloud Storage, Cloud Run) | **85% – 90%** | 3 a 5 días |
| **Google TPU Research Cloud (TRC)** | Escalado de embeddings espaciotemporales, deep learning y calibración conformal masiva | **Acceso 100% gratuito a Cloud TPUs** (v2/v3/v4) | **80% – 90%** | 3 a 7 días |

---

## 🔍 2. Estado de Verificación del Repositorio (`Incendios_PY`)

El repositorio en su versión **v2.1.0** ya cumple con creces los criterios de evaluación de Google:
- [x] **README en inglés formal y técnico:** Describe la arquitectura desacoplada (ignición biofísica primaria M2/M3 vs. propagación táctica $t-1$).
- [x] **Métricas de escala transparentes:** Malla nacional de 397.448 celdas (1 km²), 8 particiones anuales (2018–2025), 915.561.114 celda-días candidatas y 892.533 detecciones térmicas VIIRS 375 m.
- [x] **Licencia de Código Abierto:** MIT License con badge visible.
- [x] **Target Académico Explícito:** Manuscrito preparado para *Computers & Geosciences* (Elsevier).
- [x] **Suite de Pruebas Unitarias:** 76 tests pasando en CI (`pytest`).
- [x] **Integridad Criptográfica:** Builds verificados con hashes SHA-256 independientes.

---

## 📝 3. Formulario de Postulación: Google for Startups Cloud (Nivel "Start")

* **Portal de Postulación:** [https://cloud.google.com/startup](https://cloud.google.com/startup)
* **GCP Billing Account ID:** Tu cuenta de facturación vinculada a Google Cloud.
* **Correo de Contacto:** `sanchez.martin093@gmail.com`

### Campos y Respuestas Listas para Copiar/Pegar:

**Company / Project Name:**
`Incendios_PY - Open GeoAI Wildfire Early Warning System`

**Website / Product URL:**
`https://github.com/sanchezmartin093-code/Incendios_PY`

**Industry / Category:**
`Public Interest Technology / AI & Machine Learning / Environmental & Climate Tech`

**Short Project Description (Elevator Pitch):**
> *Incendios_PY is an open-source, reproducible geospatial AI framework providing daily 1 km² wildfire ignition and propagation forecasts across Paraguay and the Gran Chaco. Developed as a public-interest technology initiative, it addresses critical environmental hazard gaps in under-monitored regions by integrating NASA VIIRS 375m thermal data, ERA5-Land reanalysis, and MapBiomas land cover to protect vulnerable ecological biomes and communities.*

**How do you plan to use Google Cloud?**
> *We will utilize Google Cloud infrastructure to operationalize our open-access wildfire warning system: (1) Google Cloud Run for serverless daily forecast generation and public API endpoints; (2) Google Cloud Storage for hosting compressed geospatial Parquet matrices and historical benchmarks; and (3) Vertex AI (Gemini 3.8 Flash Multimodal) to power an automated tri-modal satellite verification loop (cross-referencing GOES-16 ABI, VIIRS 375m, and Sentinel-2 SWIR smoke penetration imagery) to deliver synthesized real-time situational bulletins to emergency responders.*

**Funding Status:**
`Bootstrapped / Independent Research Project (Pre-seed / Early stage)`

**Team Size:**
`1-5` (Seleccionar fundador individual / proyecto independiente).

---

## 📝 4. Formulario de Postulación: Google TPU Research Cloud (TRC)

* **Portal de Postulación:** [https://sites.research.google/trc/about/](https://sites.research.google/trc/about/)
* **Email:** `sanchez.martin093@gmail.com`

### Campos y Respuestas Listas para Copiar/Pegar:

**Applicant Name & Affiliation:**
`Martín Sánchez Insfrán (Independent Researcher, Geospatial AI & Open Science)`

**Project Title:**
`Incendios_PY: Decoupled Wildfire Ignition and Tactical Propagation Modeling in the Gran Chaco Using High-Resolution GeoAI`

**Code Repository:**
`https://github.com/sanchezmartin093-code/Incendios_PY`

**Publication Plans / Open Science Commitment:**
> *We have completed the core computational benchmark and prepared a full research manuscript targeted for 'Computers & Geosciences' (Elsevier). All codebase, zero-copy pipelines, deterministic evaluation scripts, and model artifacts are released publicly under the MIT Open Source License (Release v2.1.0, Zenodo concept DOI).*

**Technical Description of the Problem:**
> *Wildfires in South America (specifically the Paraguayan Chaco and Pantanal transition zones) have reached extreme severity. Incendios_PY models daily ignition and contagion across a national grid of 397,448 spatial cells at 1 km² resolution over an 8-year multi-sensor baseline (2018–2025; 915,561,114 candidate cell-days and 892,533 fire detections).*
> *The framework mathematically decouples biophysical ignition (fuel dryness, VPD, antecedent rainfall deficits) from tactical contagion at t-1. Validation incorporates multiannual out-of-time holdouts and a 31-block spatial exclusion zone (50-km blocks with 20-km dead-zone buffers) to rigorously eliminate spatial leakage.*

**Why do you need Cloud TPUs?**
> *Scaling hyperparameter exploration, spatio-temporal graph neural networks, and transformer-based climate-fuel embeddings across nearly a billion cell-days requires massive parallel tensor throughput. Furthermore, computing finite-sample conformal prediction bounds and running extensive Monte Carlo stress tests across 1.8M+ holdout cell-days exceeds consumer hardware capabilities. Cloud TPUs will accelerate this deep learning phase and allow continuous nationwide recalibration.*

---

## 💡 5. Tips Tácticos de Aprobación

1. **El argumento "Global South & High Ecological Impact":** Los evaluadores de Google priorizan investigaciones que abordan biomas críticos amenazados (Gran Chaco, Pantanal) en regiones con escasez histórica de infraestructura de datos.
2. **Cero ambigüedad comercial en TRC:** El formulario de TRC debe resaltar exclusivamente la ciencia abierta, reproducibilidad estricta y publicación en Elsevier.
3. **Rol de los créditos de Startups:** Los $2.000 USD de Startups financian la API, el almacenamiento en GCS y la inferencia multimodal de Gemini 3.8 Flash, mientras que TRC aporta el cómputo puro en aceleradores TPU sin costo alguno.
4. **Verificación de Autoría:** Si recibes un correo automático de Google Research solicitando confirmar autoría, responde de inmediato vinculando los commits firmados en el repositorio de GitHub.
