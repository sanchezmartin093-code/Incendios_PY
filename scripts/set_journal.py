#!/usr/bin/env python3
"""set_journal.py — Configura o crea el perfil editorial del manuscrito.

Carga un preset desde manuscrito/assets/perfiles_editoriales/ o crea uno nuevo
según las normas de publicación de la revista objetivo, actualizando config.yml.

Uso:
  python3 scripts/set_journal.py generic_q1
  python3 scripts/set_journal.py --list
  python3 scripts/set_journal.py accident_analysis_prevention
  python3 scripts/set_journal.py lancet_regional_health --cuerpo-max 3500 --abstract-max 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yml"
PRESETS_DIR = ROOT / "manuscrito" / "assets" / "perfiles_editoriales"


def list_presets() -> int:
    if not PRESETS_DIR.exists():
        print(f"No existe el directorio de presets: {PRESETS_DIR.relative_to(ROOT)}")
        return 0
    presets = sorted(PRESETS_DIR.glob("*.yml"))
    print(f"Presets editoriales disponibles en {PRESETS_DIR.relative_to(ROOT)}:\n")
    for p in presets:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            desc = data.get("descripcion", "Sin descripción")
            journal = data.get("journal", p.stem)
            print(f"  • {p.stem:30} [{journal}]: {desc}")
        except Exception:
            print(f"  • {p.stem}")
    print("\nPara activar uno: python3 scripts/set_journal.py <preset_nombre>")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Configurar perfil editorial de revista objetivo")
    parser.add_argument("journal", nargs="?", default=None, help="Nombre o slug del journal (ej. generic_q1, accident_analysis_prevention)")
    parser.add_argument("--list", action="store_true", help="Listar todos los perfiles editoriales disponibles")
    parser.add_argument("--fuente", default="Times New Roman", help="Fuente tipográfica (default: Times New Roman)")
    parser.add_argument("--tamano", type=int, default=12, help="Tamaño en puntos (default: 12)")
    parser.add_argument("--interlineado", type=float, default=2.0, help="Espaciado de línea (default: 2.0)")
    parser.add_argument("--margen", type=float, default=2.54, help="Margen en cm (default: 2.54)")
    parser.add_argument("--posicion-artefactos", choices=["incrustada", "final"], default="incrustada",
                        help="Posición de figuras y tablas: incrustada (default) o final")
    parser.add_argument("--abstract-min", type=int, default=150, help="Mínimo palabras abstract")
    parser.add_argument("--abstract-max", type=int, default=300, help="Máximo palabras abstract")
    parser.add_argument("--cuerpo-max", type=int, default=3500, help="Límite palabras texto principal")
    parser.add_argument("--max-items", type=int, default=5, help="Tope figuras + tablas combinadas")
    parser.add_argument("--max-refs", type=int, default=50, help="Tope de referencias bibliográficas")

    args = parser.parse_args()

    if args.list:
        return list_presets()

    if not args.journal:
        parser.print_help()
        print("\nTip: Ejecuta con --list para ver los perfiles disponibles.")
        return 1

    slug = args.journal.lower().replace(" ", "_").replace("-", "_")
    preset_file = PRESETS_DIR / f"{slug}.yml"

    if preset_file.exists():
        print(f"==> Cargando preset editorial existente: {preset_file.relative_to(ROOT)}")
        with open(preset_file, "r", encoding="utf-8") as f:
            perfil = yaml.safe_load(f)
    else:
        print(f"==> Creando nuevo perfil editorial para '{args.journal}'...")
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        perfil = {
            "journal": args.journal,
            "descripcion": f"Perfil editorial configurado para {args.journal}",
            "posicion_artefactos": args.posicion_artefactos,
            "formato_docx": {
                "fuente": args.fuente,
                "tamano_pt": args.tamano,
                "interlineado": args.interlineado,
                "margen_cm": args.margen,
                "lineas_numeradas": True,
                "tablas_apa": True,
            },
            "limites": {
                "abstract_tipo": "estructurado",
                "abstract_secciones": ["Background", "Methods", "Findings", "Interpretation"],
                "abstract_min_palabras": args.abstract_min,
                "abstract_max_palabras": args.abstract_max,
                "cuerpo_max_palabras": args.cuerpo_max,
                "max_figuras_tablas_total": args.max_items,
                "max_referencias": args.max_refs,
                "accion_limites": "advertencia",
            }
        }
        with open(preset_file, "w", encoding="utf-8") as f:
            yaml.dump(perfil, f, sort_keys=False, allow_unicode=True)
        print(f"  ✓ Preset guardado en: {preset_file.relative_to(ROOT)}")

    # Actualizar config.yml
    if not CONFIG_PATH.exists():
        print(f"❌ Error: No se encontró {CONFIG_PATH}")
        return 1

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config["editorial"] = perfil

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)

    print(f"\n✅ config.yml actualizado con el perfil de: '{args.journal}'")
    print(f"   • Fuente: {perfil['formato_docx']['fuente']} {perfil['formato_docx']['tamano_pt']}pt | Interlineado: {perfil['formato_docx']['interlineado']}")
    print(f"   • Márgenes: {perfil['formato_docx']['margen_cm']} cm | Numeración continua: {perfil['formato_docx']['lineas_numeradas']}")
    print(f"   • Posición figuras/tablas: {perfil['posicion_artefactos']}")
    print(f"   • Límites: Abstract {perfil['limites']['abstract_min_palabras']}–{perfil['limites']['abstract_max_palabras']} palabras | Cuerpo <= {perfil['limites']['cuerpo_max_palabras']} palabras")
    print(f"   • Tope artefactos: {perfil['limites']['max_figuras_tablas_total']} | Tope referencias: {perfil['limites']['max_referencias']}")
    print(f"   • Política de límites: {perfil['limites']['accion_limites']} (permite compilar y acortar luego)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
