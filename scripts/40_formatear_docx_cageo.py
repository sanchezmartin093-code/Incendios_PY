#!/usr/bin/env python3
"""Post-procesa el manuscrito para el formato de Computers & Geosciences.

Uso:
    python scripts/40_formatear_docx_cageo.py \
        entregables/manuscrito.docx \
        entregables/manuscrito_cageo.docx
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree
from PIL import Image

BLACK = RGBColor(0, 0, 0)
TNR = "Times New Roman"


def set_run_font(run, size: float | None = None, bold: bool | None = None,
                 italic: bool | None = None, color: RGBColor | None = None):
    run.font.name = TNR
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(attr), TNR)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rFonts.get(qn(attr)) is not None:
            del rFonts.attrib[qn(attr)]
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color


def set_style_font(style, size: float | None = None, bold: bool | None = None,
                   italic: bool | None = None, color: RGBColor | None = None):
    style.font.name = TNR
    rPr = style.element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(attr), TNR)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rFonts.get(qn(attr)) is not None:
            del rFonts.attrib[qn(attr)]
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    if color is not None:
        style.font.color.rgb = color
    # remove theme color and explicit italic from the style element
    if color is not None:
        for el in rPr.findall(qn("w:color")):
            rPr.remove(el)
        col = OxmlElement("w:color")
        col.set(qn("w:val"), "000000")
        rPr.insert_element_before(col, "w:spacing", "w:w", "w:kern", "w:position",
                                   "w:sz", "w:szCs", "w:highlight", "w:u", "w:effect",
                                   "w:bdr", "w:shd", "w:fitText", "w:vertAlign", "w:rtl",
                                   "w:cs", "w:em", "w:lang", "w:eastAsianLayout",
                                   "w:specVanish", "w:oMath")
    if italic is False:
        for el in rPr.findall(qn("w:i")):
            rPr.remove(el)


def set_style_paragraph(style, line: float | None = None, before: float = 0,
                        after: float = 0, align=None, keep_next: bool | None = None,
                        keep_lines: bool | None = None, left: float | None = None,
                        hanging: float | None = None):
    pf = style.paragraph_format
    if line is not None:
        pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if align is not None:
        pf.alignment = align
    if keep_next is not None:
        pf.keep_with_next = keep_next
    if keep_lines is not None:
        pf.keep_together = keep_lines
    if left is not None:
        pf.left_indent = Inches(left)
    if hanging is not None:
        pf.first_line_indent = Inches(-hanging)


def iter_paragraphs(container):
    for p in container.paragraphs:
        yield p
    for table in container.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from iter_paragraphs(cell)


def iter_tables(container):
    for table in container.tables:
        yield table
        for row in table.rows:
            for cell in row.cells:
                yield from iter_tables(cell)


def replace_in_runs(container, old: str, new: str) -> int:
    """Reemplaza old por new solo si cae dentro de un run (preserva formato)."""
    n = 0
    for p in iter_paragraphs(container):
        for run in p.runs:
            if old in run.text:
                run.text = run.text.replace(old, new)
                n += 1
    return n


def replace_paragraph_text(p, new_text: str):
    """Reemplaza todo el contenido del párrafo conservando pPr/estilo."""
    for child in list(p._p):
        if child.tag != qn("w:pPr"):
            p._p.remove(child)
    p.add_run(new_text)


def set_cell_width(cell, width_twips: int):
    tcPr = cell._tc.get_or_add_tcPr()
    for el in tcPr.findall(qn("w:tcW")):
        tcPr.remove(el)
    tcW = OxmlElement("w:tcW")
    tcW.set(qn("w:w"), str(width_twips))
    tcW.set(qn("w:type"), "dxa")
    tcPr.insert_element_before(tcW, "w:gridSpan", "w:hMerge", "w:vMerge", "w:tcBorders",
                               "w:shd", "w:noWrap", "w:tcMar", "w:textDirection",
                               "w:tcFitText", "w:vAlign", "w:hideMark")


def fix_captions(doc):
    for p in iter_paragraphs(doc):
        txt = p.text.strip()
        if not re.match(r"^(Figure|Table)[\s\u00A0]*\d+:", txt):
            continue
        pPrs = p._p.findall(qn("w:pPr"))
        if len(pPrs) > 1:
            for extra in pPrs[:-1]:
                p._p.remove(extra)
        if txt.startswith("Table"):
            p.style = doc.styles["Table Caption"]
        else:
            p.style = doc.styles["Image Caption"]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.0
        if txt.startswith("Table"):
            p.paragraph_format.keep_with_next = True
        for run in p.runs:
            set_run_font(run, size=10, italic=False, color=BLACK)


def set_internal_hyperlink_color(doc):
    body = doc.element.body
    for link in body.iter(qn("w:hyperlink")):
        if link.get(qn("w:anchor")) is None:
            continue
        for run in link.iter(qn("w:r")):
            rPr = run.find(qn("w:rPr"))
            if rPr is None:
                rPr = OxmlElement("w:rPr")
                run.insert(0, rPr)
            for el in rPr.findall(qn("w:color")):
                rPr.remove(el)
            col = OxmlElement("w:color")
            col.set(qn("w:val"), "000000")
            rPr.insert_element_before(col, "w:spacing", "w:w", "w:kern", "w:position",
                                      "w:sz", "w:szCs", "w:highlight", "w:u", "w:effect",
                                      "w:bdr", "w:shd", "w:fitText", "w:vertAlign", "w:rtl",
                                      "w:cs", "w:em", "w:lang", "w:eastAsianLayout",
                                      "w:specVanish", "w:oMath")
            for el in rPr.findall(qn("w:u")):
                rPr.remove(el)


def set_section_format(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.header_distance = Inches(0.5)
    section.footer_distance = Inches(0.5)

    # Footer with PAGE field
    footer = section.footer
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0]
    for r in list(fp.runs):
        r._element.getparent().remove(r._element)
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    set_run_font(run, size=10, color=BLACK)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    txt = OxmlElement("w:t")
    txt.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    for el in (fld_begin, instr, fld_sep, txt, fld_end):
        run._r.append(el)

    sectPr = section._sectPr
    for tag in ("w:lnNumType", "w:pgNumType"):
        for el in sectPr.findall(qn(tag)):
            sectPr.remove(el)
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), "1")
    cols.set(qn("w:space"), "720")

    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:restart"), "continuous")
    cols.addprevious(ln)
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:start"), "1")
    cols.addprevious(pg)


def format_tables(doc):
    for table in iter_tables(doc):
        nrows = len(table.rows)
        ncols = len(table.columns)
        if nrows <= 1 and ncols <= 1:
            continue  # wrapper de figura/caption

        # Ancho total 6.5 in = 9360 twips
        tbl = table._tbl
        tblPr = tbl.tblPr
        for el in tblPr.findall(qn("w:tblW")):
            tblPr.remove(el)
        tblW = OxmlElement("w:tblW")
        tblW.set(qn("w:type"), "pct")
        tblW.set(qn("w:w"), "5000")
        tblPr.insert_element_before(tblW, "w:jc", "w:tblCellSpacing", "w:tblInd",
                                    "w:tblBorders", "w:shd", "w:tblLayout",
                                    "w:tblCellMar", "w:tblLook", "w:tblCaption",
                                    "w:tblDescription")
        if tblPr.find(qn("w:tblLayout")) is None:
            layout = OxmlElement("w:tblLayout")
            layout.set(qn("w:type"), "fixed")
            tblPr.insert_element_before(layout, "w:tblCellMar", "w:tblLook",
                                        "w:tblCaption", "w:tblDescription")

        # Reparto de columnas
        header = [c.text.strip() for c in table.rows[0].cells]
        if ncols == 10 and any("ΔROC" in h for h in header) and any("Cutoff" in h for h in header):
            widths = [936, 2146, 835, 749, 893, 691, 835, 835, 749, 691]
        elif ncols == 7 and "FP" in header:
            widths = [1224, 1728, 2232, 1080, 1152, 1080, 864]
        elif ncols == 6 and any("ΔROC" in h for h in header):
            widths = [1440, 2016, 1440, 1152, 1440, 1872]
        elif ncols == 5:
            widths = [1440, 2160, 1872, 1872, 2016]
        else:
            base = 9360 // ncols
            widths = [base] * ncols
            widths[-1] += 9360 - sum(widths)

        grid = tbl.tblGrid
        cols = grid.findall(qn("w:gridCol"))
        for col, w in zip(cols, widths):
            col.set(qn("w:w"), str(w))
        for row in table.rows:
            for idx, cell in enumerate(row.cells):
                if idx < len(widths):
                    set_cell_width(cell, widths[idx])

        # Bordes: superior/inferior + bajo encabezado; sin verticales
        for el in tblPr.findall(qn("w:tblBorders")):
            tblPr.remove(el)
        borders = OxmlElement("w:tblBorders")
        for edge, val, sz in (("top", "single", "4"), ("left", "nil", "0"),
                              ("bottom", "single", "4"), ("right", "nil", "0"),
                              ("insideH", "nil", "0"), ("insideV", "nil", "0")):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), val)
            el.set(qn("w:sz"), sz)
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "000000")
            borders.append(el)
        tblPr.insert_element_before(borders, "w:shd", "w:tblLayout", "w:tblCellMar",
                                    "w:tblLook", "w:tblCaption", "w:tblDescription")

        # Filas: no partir; encabezado repetido
        for ridx, row in enumerate(table.rows):
            trPr = row._tr.get_or_add_trPr()
            if trPr.find(qn("w:cantSplit")) is None:
                trPr.insert_element_before(OxmlElement("w:cantSplit"), "w:trHeight",
                                           "w:tblHeader", "w:tblCellSpacing",
                                           "w:jc", "w:hidden")
            if ridx == 0 and trPr.find(qn("w:tblHeader")) is None:
                trPr.insert_element_before(OxmlElement("w:tblHeader"), "w:tblCellSpacing",
                                           "w:jc", "w:hidden")

        # Texto de tabla: TNR 9, interlineado simple
        for ridx, row in enumerate(table.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(0)
                    for run in p.runs:
                        set_run_font(run, size=9, color=BLACK,
                                     bold=True if ridx == 0 else None)


def anonymize(doc):
    # Data availability
    for p in doc.paragraphs:
        if p.text.startswith("Processed data, test predictions"):
            replace_paragraph_text(
                p,
                "Processed data, test predictions, trained models and derived tables are "
                "available in the anonymized review repository under CC BY 4.0 and are "
                "documented under the data concept DOI provided on the title page. Raw NASA "
                "FIRMS, ERA5-Land/CDS, MapBiomas, Hansen GFC and geoBoundaries data are "
                "obtained from their original providers under their own terms and are "
                "documented with SHA-256 hashes in fuentes/MANIFIESTO_DATOS_v3.csv.",
            )
        elif p.text.startswith("Name: Incendios_PY"):
            replace_paragraph_text(
                p,
                "Name: Incendios_PY\n"
                "Developer and contact: anonymized for peer review; contact via the Editor-in-Chief\n"
                "Year first available: 2026\n"
                "Hardware: x86_64 or ARM64, at least 8 GB RAM\n"
                "Operating system: macOS 13+, Linux Ubuntu 22.04+, Windows 11 with WSL2\n"
                "Language: Python 3.12+; Polars, DuckDB and NumPy for vectorized operations; "
                "LightGBM C++ backend\n"
                "Dependencies: pinned in requirements-lock.txt\n"
                "License: MIT\n"
                "Repository: anonymized review repository (link provided in the submission system)\n"
                "Code version DOI: provided on the title page\n"
                "Data version DOI: provided on the title page\n"
                "Code concept DOI: provided on the title page\n"
                "Data concept DOI: provided on the title page\n"
                "User guide: docs/USER_GUIDE.md\n"
                "Tutorial: docs/TUTORIAL.md\n"
                "Tests: 76 tests pass with python -m pytest -q tests.",
            )
        elif p.text.startswith("Martín Sánchez Insfrán: conceptualization"):
            replace_paragraph_text(
                p,
                "Author 1: conceptualization, methodology, software, data curation, formal "
                "analysis, validation, visualization, writing—original draft, writing—review "
                "and editing, project administration.",
            )


def replace_media(docx_path: str, mapping: dict[str, str]):
    docx_path = Path(docx_path)
    tmp = docx_path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(docx_path, "r") as zin:
        entries = {item.filename: zin.read(item.filename) for item in zin.infolist()}

    rid_to_path = {Path(k).stem: v for k, v in mapping.items()}
    for name, path in mapping.items():
        entries[name] = Path(path).read_bytes()

    # Ajustar la altura de cada imagen al nuevo aspect ratio, manteniendo el ancho
    root = etree.fromstring(entries["word/document.xml"])
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
          "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
          "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    for drawing in root.iter("{%s}drawing" % ns["w"]):
        blip = drawing.find(".//{%s}blip" % ns["a"])
        if blip is None:
            continue
        rid = blip.get("{%s}embed" % ns["r"])
        path = rid_to_path.get(rid)
        if not path:
            continue
        with Image.open(path) as im:
            img_w, img_h = im.size
        extent = drawing.find(".//{%s}extent" % ns["wp"])
        if extent is None:
            continue
        cx = int(extent.get("cx"))
        cy = int(cx * img_h / img_w)
        extent.set("cy", str(cy))
        aext = drawing.find(".//{%s}ext" % ns["a"])
        if aext is not None:
            aext.set("cy", str(cy))

    entries["word/document.xml"] = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in entries.items():
            zout.writestr(name, data)
    shutil.move(str(tmp), str(docx_path))



ORDER = {
    "tblPr": ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
              "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd",
              "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
              "tblCaption", "tblDescription"],
    "trPr": ["cnfStyle", "divId", "gridBefore", "gridAfter", "wBefore", "wAfter",
             "cantSplit", "trHeight", "tblHeader", "tblCellSpacing", "jc", "hidden"],
    "tcPr": ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd",
             "noWrap", "tcMar", "textDirection", "tcFitText", "vAlign", "hideMark"],
    "pPr": ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr",
            "widowControl", "numPr", "suppressLineNumbers", "pBdr", "shd", "tabs",
            "suppressAutoHyphens", "kinsoku", "wordWrap", "overflowPunct",
            "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
            "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents",
            "suppressOverlap", "jc", "textDirection", "textAlignment",
            "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr",
            "pPrChange"],
    "rPr": ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike",
            "dstrike", "outline", "shadow", "emboss", "imprint", "noProof",
            "snapToGrid", "vanish", "webHidden", "color", "spacing", "w", "kern",
            "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd",
            "fitText", "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout",
            "specVanish", "oMath"],
    "sectPr": ["headerReference", "footerReference", "footnotePr", "endnotePr", "type",
               "pgSz", "pgMar", "paperSrc", "pgBorders", "lnNumType", "pgNumType",
               "cols", "formProt", "vAlign", "noEndnote", "titlePg", "textDirection",
               "bidi", "rtlGutter", "docGrid", "printerSettings", "sectPrChange"],
}
NESTED = {
    "tblBorders": ["top", "left", "bottom", "right", "insideH", "insideV"],
    "tcBorders": ["top", "start", "left", "bottom", "end", "right", "insideH", "insideV", "tl2br", "tr2bl"],
}


def normalize_ooxml_order(doc):
    for local, order in ORDER.items():
        for el in doc.element.body.iter(qn("w:" + local)):
            children = list(el)
            children.sort(key=lambda c: order.index(c.tag.split("}")[-1])
                          if c.tag.split("}")[-1] in order else len(order))
            for child in children:
                el.append(child)
    for local, order in NESTED.items():
        for el in doc.element.body.iter(qn("w:" + local)):
            children = list(el)
            children.sort(key=lambda c: order.index(c.tag.split("}")[-1])
                          if c.tag.split("}")[-1] in order else len(order))
            for child in children:
                el.append(child)



def main(in_path: str, out_path: str):
    doc = Document(in_path)
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.core_properties.comments = ""

    # 1. Quitar fecha y nota interna del manuscrito
    for p in list(doc.paragraphs):
        t = p.text.strip()
        if p.style.name == "Date" or t.startswith("Target journal:") or t.startswith("Peer review:"):
            p._element.getparent().remove(p._element)

    # 2. Estilos
    set_style_font(doc.styles["Normal"], size=12, color=BLACK)
    set_style_paragraph(doc.styles["Normal"], line=2.0, before=0, after=0)

    for name in ("Body Text", "First Paragraph", "Compact", "Block Text"):
        st = doc.styles[name]
        set_style_font(st, size=12, color=BLACK)
        set_style_paragraph(st, line=2.0, before=0, after=0)

    h1 = doc.styles["Heading 1"]
    set_style_font(h1, size=12, bold=True, color=BLACK)
    set_style_paragraph(h1, line=2.0, before=12, after=6, keep_next=True)

    h2 = doc.styles["Heading 2"]
    set_style_font(h2, size=12, bold=True, color=BLACK)
    set_style_paragraph(h2, line=2.0, before=10, after=4, keep_next=True)

    title = doc.styles["Title"]
    set_style_font(title, size=17, bold=True, color=BLACK)
    set_style_paragraph(title, line=2.0, before=0, after=6,
                        align=WD_ALIGN_PARAGRAPH.CENTER)

    subtitle = doc.styles["Subtitle"]
    set_style_font(subtitle, size=12, italic=True, color=BLACK)
    set_style_paragraph(subtitle, line=2.0, before=0, after=6,
                        align=WD_ALIGN_PARAGRAPH.CENTER)

    for name in ("Caption", "Image Caption", "Table Caption"):
        st = doc.styles[name]
        set_style_font(st, size=10, italic=False, color=BLACK)
        set_style_paragraph(st, line=1.0, before=6, after=6)

    bib = doc.styles["Bibliography"]
    set_style_font(bib, size=12, color=BLACK)
    set_style_paragraph(bib, line=2.0, before=0, after=0, left=0.5, hanging=0.5)

    for name in ("Abstract", "Abstract Title", "Date", "Author"):
        if name in [s.name for s in doc.styles]:
            st = doc.styles[name]
            set_style_font(st, size=12, color=BLACK)

    # 3. Sección, líneas y paginación
    set_section_format(doc)

    # 4. Captions y tablas
    fix_captions(doc)
    format_tables(doc)

    # 5. Contenido: anonimato y consistencia
    anonymize(doc)

    replacements = [
        ("7. Acknowledgements", "Acknowledgments"),
        ("7. Acknowledgments", "Acknowledgments"),
        ("summarises", "summarizes"),
        ("prioritises", "prioritizes"),
        ("prioritisation", "prioritization"),
        ("labour", "labor"),
        ("Acknowledgements", "Acknowledgments"),
        ("ΔROC vs M3", "ΔROC (M3 − reference)"),
        ("32.81 % versus 33.27 %", "32.82 % versus 33.28 %"),
        ("National 1-km² alert map", "Illustrative national 1-km² alert map"),
        ("over the national 1-km² grid.", "over the national 1-km² grid (EPSG:32721)."),
        ("for full Canadian FWI", "for the full Canadian Forest Fire Weather Index (FWI)"),
        ("for INPE;", "for the INPE Risco de Fogo;"),
        ("the full Canadian FWI and 0.5389/0.4990 for INPE",
         "the full Canadian Forest Fire Weather Index (FWI) and 0.5389/0.4990 for the INPE Risco de Fogo"),
        ("VPD below 0.8 kPa", "vapor pressure deficit (VPD) below 0.8 kPa"),
        ("FFMC and ISI", "Fine Fuel Moisture Code (FFMC) and Initial Spread Index (ISI)"),
        ("accumulated FRP", "accumulated fire radiative power (FRP)"),
        ("Hansen GFC-2025 v1.13", "Hansen Global Forest Change (GFC-2025 v1.13)"),
        ("UTM zone 21S", "Universal Transverse Mercator (UTM) zone 21S"),
        ("The primary metrics are ROC-AUC and PR-AUC.",
         "The primary metrics are the area under the receiver operating characteristic curve (ROC-AUC) and precision–recall area under the curve (PR-AUC)."),
        ("F0.5 criterion", "F0.5 (β = 0.5) criterion"),
        ("VIIRS record", "Visible Infrared Imaging Radiometer Suite (VIIRS) record"),
        ("Instituto Nacional de Pesquisas Espaciais (INPE) Risco de Fogo", "INPE Risco de Fogo"),
        ("6. Data availability", "Data availability"),
        ("7. Acknowledgements", "Acknowledgments"),
        ("8. Funding", "Funding"),
        ("9. Competing interests", "Competing interests"),
        ("10. Author contributions", "Author contributions"),
        ("11. Generative AI declaration", "Generative AI declaration"),
        ("12. Computer code availability", "Computer Code Availability"),
        # Referencias
        ("Groot, W. J. de", "de Groot, W. J."),
        ("Silva Menezes, L. da", "da Silva Menezes, L."),
        ("Pelissier", "Pélissier"),
        ("Piquer-Rodrı́guez", "Piquer-Rodríguez"),
        ("southern amazon", "southern Amazon"),
        ("the chaco", "the Chaco"),
        ("dry chaco", "dry Chaco"),
        ("the canadian forest fire weather index system", "The Canadian Forest Fire Weather Index System"),
        ("brazil’s", "Brazil’s"),
        ("the european", "the European"),
        ("europe:", "Europe:"),
        ("european", "European"),
        ("The canadian", "The Canadian"),
        ("detroit", "Detroit"),
        ("ansi-c", "ANSI-C"),
        ("s: the gstat", "S: the gstat"),
    ]
    for old, new in replacements:
        replace_in_runs(doc, old, new)

    # Definir INPE una sola vez, en la introducción, no en cada celda de tabla
    for p in doc.paragraphs:
        if p.text.startswith("Vegetation fires"):
            for run in p.runs:
                if "and the INPE Risco de Fogo" in run.text:
                    run.text = run.text.replace(
                        "and the INPE Risco de Fogo",
                        "and the Instituto Nacional de Pesquisas Espaciais (INPE) Risco de Fogo",
                    )
                    break

    # Definir ROC-AUC solo en el abstract (no redefinir la sigla en Conclusions)
    for p in doc.paragraphs:
        if p.text.startswith("Wildfire machine learning"):
            for run in p.runs:
                if "ROC-AUC 0.7844/0.7628" in run.text:
                    run.text = run.text.replace(
                        "ROC-AUC 0.7844/0.7628",
                        "an area under the receiver operating characteristic curve "
                        "(ROC-AUC) of 0.7844/0.7628",
                    )
                    break

    # 6. Miles con espacio no separable (solo texto/tablas, no URLs/DOIs)
    for p in iter_paragraphs(doc):
        for run in p.runs:
            if run._element.findall(".//" + qn("w:drawing")):
                continue
            if "doi.org" in run.text or "http" in run.text:
                continue
            run.text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "\u00A0", run.text)

    # 7. Enlaces internos en negro
    set_internal_hyperlink_color(doc)

    # 8. Ordenar hijos de propiedades OOXML según esquema (evita reparación en Word)
    normalize_ooxml_order(doc)

    doc.save(out_path)
    figdir = Path("entregables/figuras")
    replace_media(out_path, {
        "word/media/rId12.png": str(figdir / "fig01_framework_conceptual.png"),
        "word/media/rId18.png": str(figdir / "fig01_macro_regiones_cobertura.png"),
        "word/media/rId26.png": str(figdir / "fig02_curvas_roc_pr_test.png"),
        "word/media/rId32.png": str(figdir / "fig05_benchmarking_curvas_roc_pr.png"),
        "word/media/rId39.png": str(figdir / "fig03_mapa_riesgo_operativo_nacional.png"),
    })
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Uso: python scripts/40_formatear_docx_cageo.py INPUT OUTPUT")
    main(sys.argv[1], sys.argv[2])
