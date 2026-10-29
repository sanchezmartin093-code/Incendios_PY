# Deliverables

Navigation index for the final products of `Incendios_PY`. All files are
regenerated from `manuscrito/` and `scripts/`; do not edit them by hand.

## Structure

| Folder | Contents |
|---|---|
| `en/` | Official English submission version (DOCX): manuscript, supplementary material, cover letter, highlights and navigable supplement. |
| `es/` | Spanish reading version (DOCX): manuscript, supplementary material and title page. |
| `maestros/` | Master artifacts: `figuras_maestro.pdf` (one figure per page), `tablas_maestro.xlsx` (one sheet per table) and `graphical_abstract.png`. |
| `fuentes/figuras/` | Individual source figures, high-resolution PNG only. |
| `fuentes/tablas/` | Individual source tables, CSV only. |
| `web/` | Self-contained HTML previews and interactive dashboard. |

## Regeneration

```bash
make render     # manuscript and supplement DOCX + HTML
make maestros   # figuras_maestro.pdf, tablas_maestro.xlsx and graphical abstract
make submission # editorial package in dist/submission/
```

## Notes

- The submission package (`dist/submission/`) is temporary, Git-ignored and built
  on demand with `make submission`.
- Free-standing figures are released as `.png` only; tables are released as `.csv`
  only.
- Exceptions are the compiled artifacts in `maestros/` and the DOCX/HTML files in
  `en/`, `es/` and `web/`.
