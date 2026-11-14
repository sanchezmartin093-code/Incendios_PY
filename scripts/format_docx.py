#!/usr/bin/env python3
"""format_docx.py — Post-procesador editorial universal para archivos Word (.docx).

Aplica automáticamente el perfil editorial definido en config.yml:
1. Inyección de numeración continua de líneas (continuous line numbering) para revisores.
2. Normalización de fuentes (ej. Times New Roman / Arial) y eliminación de fuentes temáticas de Word.
3. Ajuste estricto de interlineado (ej. 2.0 doble espacio) y márgenes (2.54 cm).
4. Formateo de tablas estilo APA/editorial (sin líneas verticales, 3 horizontales).

Uso:
  python3 scripts/format_docx.py entregables/en/manuscrito_en.docx
"""

from __future__ import annotations

import sys
from pathlib import Path

# Garantizar que el directorio scripts/ esté en sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import yaml
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

try:
    from normalize_docx import normalize_docx
except ImportError:
    def normalize_docx(path):
        pass

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yml"


def load_editorial_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            if "editorial" in cfg:
                return cfg["editorial"]
    # Fallback predeterminado Q1
    return {
        "formato_docx": {
            "fuente": "Times New Roman",
            "tamano_pt": 12,
            "interlineado": 2.0,
            "margen_cm": 2.54,
            "lineas_numeradas": True,
            "tablas_apa": True,
        }
    }


def set_run_font(run, font_name: str, size_pt: float, bold: bool | None = None, italic: bool | None = None, color: RGBColor | None = None):
    run.font.name = font_name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(attr), font_name)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rFonts.get(qn(attr)) is not None:
            del rFonts.attrib[qn(attr)]
    if size_pt is not None:
        run.font.size = Pt(size_pt)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color


def format_paragraph(p, font_name: str, base_size: float, base_line_spacing: float):
    style_name = (p.style.name if p.style else "").lower()

    # Título principal del manuscrito
    if "title" in style_name:
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(12)
        for r in p.runs:
            set_run_font(r, font_name, base_size + 4, bold=True)
        return

    # Subtítulo
    if "subtitle" in style_name:
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(8)
        for r in p.runs:
            set_run_font(r, font_name, base_size + 1, italic=True)
        return

    # Encabezados de nivel 1 (# 1. Introduction)
    if "heading 1" in style_name:
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        for r in p.runs:
            set_run_font(r, font_name, base_size + 2, bold=True)
        return

    # Encabezados de nivel 2 (## 2.1 Methods)
    if "heading 2" in style_name:
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        for r in p.runs:
            set_run_font(r, font_name, base_size + 1, bold=True)
        return

    # Encabezados de nivel 3 (### Subsección)
    if "heading 3" in style_name:
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        for r in p.runs:
            set_run_font(r, font_name, base_size, bold=True, italic=True)
        return

    # Epígrafes de figuras y tablas (captions)
    if "caption" in style_name or ("table" in style_name and "title" in style_name):
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        for r in p.runs:
            set_run_font(r, font_name, max(base_size - 2, 9.0))
        return

    # Texto normal / cuerpo del manuscrito
    p.paragraph_format.line_spacing = base_line_spacing
    for r in p.runs:
        is_bold = r.font.bold
        is_italic = r.font.italic
        set_run_font(r, font_name, base_size, bold=is_bold, italic=is_italic)


def apply_line_numbering_and_margins(doc: Document, enable_ln: bool, margin_cm: float):
    margin_in = margin_cm / 2.54
    for section in doc.sections:
        # Márgenes
        section.top_margin = Inches(margin_in)
        section.bottom_margin = Inches(margin_in)
        section.left_margin = Inches(margin_in)
        section.right_margin = Inches(margin_in)

        # Numeración de líneas
        sectPr = section._sectPr
        for child in list(sectPr):
            if child.tag.endswith("lnNumType"):
                sectPr.remove(child)

        if enable_ln:
            lnNumType = OxmlElement("w:lnNumType")
            lnNumType.set(qn("w:countBy"), "1")
            lnNumType.set(qn("w:restart"), "continuous")
            sectPr.insert_element_before(
                lnNumType,
                "w:pgNumType",
                "w:cols",
                "w:formProt",
                "w:vAlign",
                "w:noEndnote",
                "w:titlePg",
                "w:textDirection",
                "w:bidi",
                "w:rtlGutter",
                "w:docGrid",
                "w:printerSettings",
                "w:sectPrChange",
            )


def add_page_numbers(doc: Document) -> None:
    """Insert a centered PAGE field in every section footer if absent."""
    for section in doc.sections:
        footer = section.footer
        if any("PAGE" in (node.text or "") for node in footer._element.xpath(".//w:instrText")):
            continue
        paragraph = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run()
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = "PAGE"
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        run._r.append(begin)
        run._r.append(instr)
        run._r.append(end)


def apply_apa_table_borders(table):
    tblPr = table._element.xpath("w:tblPr")
    if not tblPr:
        return
    for existing in tblPr[0].xpath("w:tblBorders"):
        tblPr[0].remove(existing)
    tblBorders = OxmlElement("w:tblBorders")

    # Borde superior e inferior visibles
    top = OxmlElement("w:top")
    top.set(qn("w:val"), "single")
    top.set(qn("w:sz"), "6")
    top.set(qn("w:space"), "0")
    top.set(qn("w:color"), "000000")
    tblBorders.append(top)

    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "0")
    bottom.set(qn("w:color"), "000000")
    tblBorders.append(bottom)

    insideH = OxmlElement("w:insideH")
    insideH.set(qn("w:val"), "none")
    tblBorders.append(insideH)

    # Bordes verticales eliminados (APA)
    for border_name in ("left", "right", "insideV"):
        b = OxmlElement(f"w:{border_name}")
        b.set(qn("w:val"), "none")
        tblBorders.append(b)

    tblPr[0].insert_element_before(
        tblBorders,
        "w:shd",
        "w:tblLayout",
        "w:tblCellMar",
        "w:tblLook",
        "w:tblCaption",
        "w:tblDescription",
        "w:tblPrChange",
    )

    # APA: horizontal rule below the header row, no per-row rules.
    if table.rows:
        for cell in table.rows[0].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            for existing in tcPr.xpath("w:tcBorders"):
                tcPr.remove(existing)
            tcBorders = OxmlElement("w:tcBorders")
            header_bottom = OxmlElement("w:bottom")
            header_bottom.set(qn("w:val"), "single")
            header_bottom.set(qn("w:sz"), "6")
            header_bottom.set(qn("w:space"), "0")
            header_bottom.set(qn("w:color"), "000000")
            tcBorders.append(header_bottom)
            tcPr.insert_element_before(
                tcBorders,
                "w:shd",
                "w:noWrap",
                "w:tcMar",
                "w:textDirection",
                "w:tcFitText",
                "w:vAlign",
                "w:hideMark",
                "w:headers",
                "w:cellIns",
                "w:cellDel",
                "w:cellMerge",
                "w:tcPrChange",
            )


def is_non_data_table(table) -> bool:
    """True for Pandoc layout wrappers and figure/caption containers."""
    if (
        len(table.rows) == 1
        and len(table.columns) == 1
        and len(table.rows[0].cells[0].tables) > 0
    ):
        return True
    xml = table._element.xml
    if "<w:drawing" in xml or "<w:pict" in xml:
        return True
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                style = (paragraph.style.name if paragraph.style else "") or ""
                if "Caption" in style:
                    return True
    return False


def clear_table_borders(tbl) -> None:
    tblPr = tbl.xpath("w:tblPr")
    if not tblPr:
        return
    for existing in tblPr[0].xpath("w:tblBorders"):
        tblPr[0].remove(existing)
    borders = OxmlElement("w:tblBorders")
    for name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{name}")
        border.set(qn("w:val"), "none")
        borders.append(border)
    tblPr[0].insert_element_before(
        borders,
        "w:shd",
        "w:tblLayout",
        "w:tblCellMar",
        "w:tblLook",
        "w:tblCaption",
        "w:tblDescription",
        "w:tblPrChange",
    )
    for tc in tbl.xpath("w:tr/w:tc"):
        tcPr = tc.xpath("w:tcPr")
        if tcPr:
            for existing in tcPr[0].xpath("w:tcBorders"):
                tcPr[0].remove(existing)


def format_table_recursive(table, font_name, font_size, tablas_apa: bool) -> None:
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.line_spacing = 1.0
                for run in paragraph.runs:
                    set_run_font(run, font_name, font_size - 2)
            for nested in cell.tables:
                format_table_recursive(nested, font_name, font_size, tablas_apa)
    if tablas_apa:
        if is_non_data_table(table):
            clear_table_borders(table._element)
        else:
            apply_apa_table_borders(table)


def add_running_head(doc: Document, text: str, filename: str, font_name: str) -> None:
    fn = filename.lower()
    # Aplicar running head a manuscritos principales (en español o inglés)
    if not text or not any(k in fn for k in ("manuscrito", "manuscript", "paper", "articulo", "article")):
        return
    for section in doc.sections:
        header = section.header
        if any(text in paragraph.text for paragraph in header.paragraphs):
            continue
        paragraph = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
        paragraph.text = text
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for run in paragraph.runs:
            set_run_font(run, font_name, 9)


def format_document(input_path: Path, output_path: Path | None = None) -> None:
    if output_path is None:
        output_path = input_path

    editorial = load_editorial_config()
    fmt = editorial.get("formato_docx", {})
    font_name = fmt.get("fuente", "Times New Roman")
    font_size = fmt.get("tamano_pt", 12)
    line_spacing = fmt.get("interlineado", 2.0)
    margin_cm = fmt.get("margen_cm", 2.54)
    enable_ln = fmt.get("lineas_numeradas", True)
    tablas_apa = fmt.get("tablas_apa", True)
    running_head = str(fmt.get("running_head", ""))

    doc = Document(input_path)

    # 1. Márgenes, numeración de líneas y paginación
    apply_line_numbering_and_margins(doc, enable_ln, margin_cm)
    add_page_numbers(doc)
    add_running_head(doc, running_head, input_path.name, font_name)

    # 2. Estilo base explícito (no depender de fuentes temáticas de Word)
    normal = doc.styles["Normal"]
    normal.font.name = font_name
    normal.font.size = Pt(font_size)
    normal_rpr = normal.element.get_or_add_rPr()
    normal_rfonts = normal_rpr.get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        normal_rfonts.set(qn(attr), font_name)
    normal.paragraph_format.line_spacing = line_spacing

    # 3. Párrafos, jerarquía de encabezados e interlineado
    for p in doc.paragraphs:
        format_paragraph(p, font_name, font_size, line_spacing)

    # 4. Tablas (incluye tablas anidadas dentro de los contenedores de Pandoc)
    for table in doc.tables:
        format_table_recursive(table, font_name, font_size, tablas_apa)

    doc.save(output_path)
    normalize_docx(Path(output_path))
    print(f"  ✓ DOCX formateado exitosamente ({output_path.name}): fuente={font_name}, interlineado={line_spacing}, líneas_numeradas={enable_ln}")


def main() -> int:
    # 1. Modo con argumento de archivo directo
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("--"):
        in_file = Path(sys.argv[1])
        out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else in_file

        if not in_file.exists():
            print(f"❌ Error: Archivo no encontrado: {in_file}")
            return 1

        print(f"==> Aplicando post-procesado editorial sobre: {in_file.name}...")
        format_document(in_file, out_file)
        return 0

    # 2. Modo auto-discovery (ejecutado por hook post-render de Quarto o sin argumentos)
    targets: list[Path] = []
    search_dirs = [
        ROOT / "entregables",
        ROOT / "revision_anonima",
        ROOT / "manuscrito",
    ]
    for sdir in search_dirs:
        if sdir.exists():
            for p in sdir.rglob("*.docx"):
                # Ignorar archivos temporales o de bloqueo de Word / LibreOffice
                if p.name.startswith(("~", ".")) or "#" in p.name:
                    continue
                targets.append(p)

    if not targets:
        print("==> Post-procesado editorial: No se encontraron archivos .docx en entregables/ ni revision_anonima/.")
        return 0

    print(f"==> Post-procesado editorial automático (Quarto post-render hook):")
    print(f"    Detectados {len(targets)} archivo(s) .docx para formatear según perfil editorial activo:")
    for t in sorted(targets):
        try:
            rel = t.relative_to(ROOT)
        except ValueError:
            rel = t
        print(f"    • Procesando: {rel}...")
        format_document(t, t)

    print("  ✓ Todos los documentos Word han sido alineados con las normas de la revista.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
