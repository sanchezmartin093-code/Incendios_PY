# Checklist de envío

**Revista:** Computers & Geosciences (Elsevier)
**Manuscrito:** A reproducible computational framework for decoupled wildfire ignition and propagation benchmarking in the Gran Chaco (2018–2025)
**Versión:** v2.1.0 (2026-10-29)
**Autor:** Martín Sánchez Insfrán

## Archivos

- [x] `entregables/submission/manuscrito.docx`
- [x] `entregables/submission/material_suplementario.docx`
- [x] `entregables/submission/highlights.docx`
- [x] `entregables/submission/cover_letter.docx`
- [x] `entregables/submission/figuras_maestro.pdf`
- [x] `entregables/submission/tablas_maestro.xlsx`
- [x] `entregables/submission/figures/` con 15 figuras individuales

## Metadatos

- [x] Título, afiliación, ORCID, e-mail y teléfono sin errores.
- [x] Keywords separadas por puntos medios.
- [x] CRediT antes de Funding.
- [x] Declaración de IA antes de References.
- [x] Funding y competing interests completos.
- [x] Concept DOI de código y datos incluidos.
- [x] Graphical abstract como archivo separado; no numerado en el cuerpo.
- [x] Release de código v2.1.0 disponible en GitHub.
- [x] Asset de datos v2.0.1 disponible en GitHub (`Incendios_PY_data_v2.0.1.zip`, 1.18 GB).
- [ ] Depositar el código v2.1.0 en Zenodo; el DOI conceptual de código resuelve a v2.0.1.

## Comprobaciones

- [x] `python3 scripts/70_verificar_envio.py` devuelve `check: OK`.
- [x] `python -m pytest -q tests` devuelve 76 tests.
- [x] `auditoria/HASHES.sha256` verifica sin errores.
- [x] Figura 4 contiene `Ignition` y `Propagation`.
- [x] Ecuaciones del suplemento conservan fracciones y sumatorias.
- [x] Tablas S6, S17 y S28 tienen la nota de precisión poblacional.
- [x] Separadores de miles uniformes con espacio.
- [x] Figuras maestras: 15 páginas y 15 marcadores.
- [x] Figuras individuales: Figure_1–Figure_5 y Figure_S1–Figure_S10.
- [x] Tablas maestras: 50 hojas, incluida `Indice`.
- [x] Table 3 y tablas S6/S8/S27/S28 legibles tras el formateo DOCX.

## Revisión humana pendiente

- [ ] Abrir los DOCX en Microsoft Word y revisar ecuaciones, tablas anchas y saltos de página.
- [ ] Confirmar que el sistema editorial acepta los DOCX y los maestros como archivos complementarios.
- [ ] Depositar el código v2.1.0 manualmente en Zenodo; el depósito automático no generó un DOI de versión.
