"""
Script 17: Generador de Figura 06 Trimodal Multiespectral (C&G Elsevier Q1).
Compone los 3 paneles de teledetección operativa con espaciado editorial limpio,
garantizando que el título principal no colisione con los subtítulos de los paneles,
y exportando tanto PNG (300 DPI) como PDF vectorial.
"""

from pathlib import Path
import sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT_DIR = Path(__file__).resolve().parent.parent
DIR_FIG = ROOT_DIR / "figuras"

def generar_figura_trimodal():
    print("=" * 65)
    print("GENERANDO FIGURA 06: VERIFICACIÓN SATELITAL TRIMODAL MULTIESPECTRAL")
    print("=" * 65)

    p1 = DIR_FIG / "modo1_regional.png"
    p2 = DIR_FIG / "modo2_truecolor_viirs.png"
    p3 = DIR_FIG / "modo3_swir_viirs.png"

    if not (p1.exists() and p2.exists() and p3.exists()):
        raise FileNotFoundError("Imágenes de los modos satelitales no encontradas en figuras/")

    im1 = Image.open(p1)
    im2 = Image.open(p2)
    im3 = Image.open(p3)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"]

    fig = plt.figure(figsize=(18, 6.8), facecolor="#FFFFFF", dpi=300)

    # Margen superior generoso para separar el título general de los paneles
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.12,
                           left=0.03, right=0.97, top=0.84, bottom=0.06)

    # Cabecera global
    fig.text(0.50, 0.95, "Verificación y Monitoreo Satelital Tri-Modal Multiespectral (Incendios_PY)",
             fontsize=14, fontweight="bold", color="#0F172A", ha="center")
    fig.text(0.50, 0.90, "Cadena Operativa de Detección Térmica: Desde la Alerta Regional Predictiva hasta la Confirmación Infrarroja",
             fontsize=10.5, color="#475569", ha="center")

    paneles = [
        (gs[0, 0], im1, "Panel A: Modo 1 - Alerta Predictiva Regional (1 km²)", "Modelo Desacoplado Incendios_PY (LightGBM + Platt)"),
        (gs[0, 1], im2, "Panel B: Modo 2 - Detección Óptica VIIRS (375 m)", "Composición Color Verdadero Suomi-NPP / NOAA-20"),
        (gs[0, 2], im3, "Panel C: Modo 3 - Confirmación Espectral SWIR", "Frente Activo Infrarrojo de Onda Corta (Canal M11 / I2)")
    ]

    for gs_cell, img, titulo, subtitulo in paneles:
        ax = fig.add_subplot(gs_cell)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
            spine.set_linewidth(1.0)
        
        ax.set_title(f"{titulo}\n{subtitulo}", fontsize=9.2, fontweight="bold", color="#1E293B", pad=8)

    p_png = DIR_FIG / "fig06_comparativa_satelital_tres_modos.png"
    p_pdf = DIR_FIG / "fig06_comparativa_satelital_tres_modos.pdf"

    fig.savefig(p_png, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(p_pdf, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Figura 06 PNG exportada: {p_png}")
    print(f"✓ Figura 06 PDF exportada: {p_pdf}")
    print("=" * 65)

if __name__ == "__main__":
    generar_figura_trimodal()
