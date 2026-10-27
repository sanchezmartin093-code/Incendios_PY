"""
Script 11: Alert Exporter to Standard GeoTIFF Raster (EPSG:32721).
- Converts the daily alert tabular Parquet layer into a multi-band GeoTIFF ready for QGIS/ArcGIS.
- Band 1: Continuous probability P(Fire|X) ∈ [0.0, 1.0].
- Band 2: Operational risk category (0 to 4).
- Output location: datos/outputs/raster_riesgo_diario/alerta_nacional_{fecha}.tif
"""

from pathlib import Path
import sys
import argparse

# Ensure the project root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constantes import DIR_OUTPUTS
from src.raster import exportar_alerta_a_geotiff

def main():
    parser = argparse.ArgumentParser(description="Exporta alerta diaria Parquet a GeoTIFF estándar.")
    parser.add_argument("--fecha", type=str, default=None, help="Fecha en formato YYYY-MM-DD")
    args = parser.parse_args()

    dir_raster = DIR_OUTPUTS / "raster_riesgo_diario"
    
    if args.fecha:
        p_parquet = dir_raster / f"alerta_nacional_{args.fecha}.parquet"
    else:
        archivos = sorted(list(dir_raster.glob("alerta_nacional_*.parquet")))
        if not archivos:
            raise FileNotFoundError("No se encontraron archivos de alerta en datos/outputs/raster_riesgo_diario/")
        p_parquet = archivos[-1]

    fecha = p_parquet.stem.replace("alerta_nacional_", "")
    p_salida_tif = dir_raster / f"alerta_nacional_{fecha}.tif"

    print("=" * 65)
    print("EXPORTACIÓN A RÁSTER GEOTIFF ESTÁNDAR (EPSG:32721)")
    print("=" * 65)
    print(f"Insumo Parquet: {p_parquet.name}")
    print(f"Destino GeoTIFF: {p_salida_tif.name}")

    exportar_alerta_a_geotiff(p_parquet, p_salida_tif)

    tamanio_kb = p_salida_tif.stat().st_size / 1024
    print(f"\n✓ GeoTIFF generado con éxito: {p_salida_tif.name} ({tamanio_kb:.1f} KB)")
    print(f"✓ Bandas: [1] Probabilidad de Fuego, [2] Nivel de Alerta Operativo (0-4)")
    print(f"✓ Resolución: 1000m x 1000m en UTM Zona 21 Sur (EPSG:32721)")
    print("=" * 65)

if __name__ == "__main__":
    main()
