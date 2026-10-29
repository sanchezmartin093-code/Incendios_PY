# Checklist de envío

**Revista:** Computers & Geosciences (Elsevier)
**Manuscrito:** A reproducible computational framework for decoupled wildfire ignition and propagation benchmarking in the Gran Chaco (2018–2025)
**Versión:** v2.1.0 (2026-10-28)
**Autor:** Martín Sánchez Insfrán

## Archivos

- [ ] `entregables/submission/manuscrito.docx`
- [ ] `entregables/submission/material_suplementario.docx`
- [ ] `entregables/submission/highlights.docx`
- [ ] `entregables/submission/cover_letter.docx`
- [ ] `entregables/submission/figuras_maestro.pdf`
- [ ] `entregables/submission/tablas_maestro.xlsx`

## Metadatos

- [ ] Título, afiliación, ORCID, e-mail y teléfono sin errores.
- [ ] Keywords separadas por puntos medios.
- [ ] CRediT antes de Funding.
- [ ] Declaración de IA antes de References.
- [ ] Funding y competing interests completos.
- [ ] Concept DOI de código y datos incluidos.
- [ ] Release v2.1.0 disponible en GitHub.
- [ ] DOI conceptual resuelve a la versión más reciente depositada.

## Comprobaciones

- [ ] `python3 scripts/70_verificar_envio.py` devuelve `check: OK`.
- [ ] `python -m pytest -q tests` devuelve 76 tests.
- [ ] `auditoria/HASHES.sha256` verifica sin errores.
- [ ] Figura 4 contiene `Ignition` y `Propagation`.
- [ ] Ecuaciones del suplemento conservan fracciones y sumatorias.
- [ ] Tablas S6, S17 y S28 tienen la nota de precisión poblacional.
- [ ] Separadores de miles uniformes con espacio.
- [ ] Figuras maestras: 15 páginas y 15 marcadores.
- [ ] Tablas maestras: 50 hojas, incluida `Indice`.

## Revisión humana pendiente

- [ ] Abrir los DOCX en Microsoft Word y revisar ecuaciones, tablas anchas y saltos de página.
- [ ] Confirmar que el sistema editorial acepta los DOCX y los maestros como archivos complementarios.
- [ ] Confirmar el DOI de versión en Zenodo si el depósito automático lo genera.
