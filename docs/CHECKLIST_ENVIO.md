# Checklist de envío

**Revista:** Computers & Geosciences (Elsevier)
**Manuscrito:** A reproducible computational framework for decoupled wildfire ignition and propagation benchmarking in the Gran Chaco (2018–2025)
**Versión:** v2.1.0 (2026-10-28)
**Autor:** Martín Sánchez Insfrán

## Archivos

- [x] `entregables/submission/manuscrito.docx`
- [x] `entregables/submission/material_suplementario.docx`
- [x] `entregables/submission/highlights.docx`
- [x] `entregables/submission/cover_letter.docx`
- [x] `entregables/submission/figuras_maestro.pdf`
- [x] `entregables/submission/tablas_maestro.xlsx`

## Metadatos

- [x] Título, afiliación, ORCID, e-mail y teléfono sin errores.
- [x] Keywords separadas por puntos medios.
- [x] CRediT antes de Funding.
- [x] Declaración de IA antes de References.
- [x] Funding y competing interests completos.
- [x] Concept DOI de código y datos incluidos.
- [x] Release v2.1.0 disponible en GitHub.
- [ ] Depositar v2.1.0 en Zenodo; actualmente el DOI conceptual resuelve a v2.0.1.

## Comprobaciones

- [x] `python3 scripts/70_verificar_envio.py` devuelve `check: OK`.
- [x] `python -m pytest -q tests` devuelve 76 tests.
- [x] `auditoria/HASHES.sha256` verifica sin errores.
- [x] Figura 4 contiene `Ignition` y `Propagation`.
- [x] Ecuaciones del suplemento conservan fracciones y sumatorias.
- [x] Tablas S6, S17 y S28 tienen la nota de precisión poblacional.
- [x] Separadores de miles uniformes con espacio.
- [x] Figuras maestras: 15 páginas y 15 marcadores.
- [x] Tablas maestras: 50 hojas, incluida `Indice`.

## Revisión humana pendiente

- [ ] Abrir los DOCX en Microsoft Word y revisar ecuaciones, tablas anchas y saltos de página.
- [ ] Confirmar que el sistema editorial acepta los DOCX y los maestros como archivos complementarios.
- [ ] Confirmar el DOI de versión en Zenodo si el depósito automático lo genera.
