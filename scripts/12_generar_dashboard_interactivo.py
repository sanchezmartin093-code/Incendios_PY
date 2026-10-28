"""
Script 12: Interactive Early Warning Web Dashboard Generator (Leaflet + Chart.js).
- Builds a standalone interactive web portal (standalone HTML): entregables/dashboard_alerta_interactivo.html
- Integrates:
  * Leaflet map with official departmental boundaries and visualization of critical hotspots.
  * Interactive circular markers colored by risk level with biophysical popups.
  * Side panel with national KPIs and a breakdown by department.
  * Interactive Chart.js charts with the distribution of areas under alert.
- Requires no server or backend: runs directly in the user's browser or can be published on GitHub Pages.
"""

from pathlib import Path
import sys
import json
import argparse
import polars as pl
import geopandas as gpd

# Ensure the project root is on the PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constantes import DIR_OUTPUTS, DIR_RAW

def generar_dashboard_interactivo(fecha: str | None = None) -> Path:
    print("=" * 65)
    print("GENERADOR DE DASHBOARD WEB INTERACTIVO (LEAFLET + CHART.JS)")
    print("=" * 65)

    dir_raster = DIR_OUTPUTS / "raster_riesgo_diario"
    if fecha:
        p_parquet = dir_raster / f"alerta_nacional_{fecha}.parquet"
        p_json = DIR_OUTPUTS / f"boletin_alerta_{fecha}.json"
    else:
        archivos = sorted(list(dir_raster.glob("alerta_nacional_*.parquet")))
        if not archivos:
            raise FileNotFoundError("No se encontraron alertas en datos/outputs/raster_riesgo_diario/")
        p_parquet = archivos[-1]
        fecha = p_parquet.stem.replace("alerta_nacional_", "")
        p_json = DIR_OUTPUTS / f"boletin_alerta_{fecha}.json"

    if not p_json.exists():
        raise FileNotFoundError(f"Boletín JSON no encontrado en {p_json}")

    with open(p_json, "r", encoding="utf-8") as f:
        resumen = json.load(f)

    # Load departmental GeoJSON boundaries and simplify for web performance
    path_adm1 = DIR_RAW / "vector" / "paraguay_adm1_departamentos.geojson"
    gdf_deptos = gpd.read_file(path_adm1)
    gdf_deptos_simple = gdf_deptos.copy()
    gdf_deptos_simple["geometry"] = gdf_deptos.geometry.simplify(0.005, preserve_topology=True)
    deptos_geojson = json.loads(gdf_deptos_simple.to_json())

    # Load highest-risk cells (Moderate, High, and Extreme) for map visualization
    df_alerta = pl.read_parquet(p_parquet)
    df_criticos = df_alerta.filter(pl.col("nivel_alerta").is_in(["MODERADO", "ALTO", "EXTREMO"]))

    # Subsample if there are too many cells to keep the HTML lightweight (< 2000 markers)
    if df_criticos.height > 1500:
        step = max(1, df_criticos.height // 1500)
        df_plot = df_criticos.gather_every(step)
    else:
        df_plot = df_criticos

    puntos_mapa = df_plot.select([
        "cell_id", "lat", "lon", "departamento", "region", "probabilidad_fuego",
        "nivel_alerta", "isi", "vpd_max_kpa", "viento_max_kmh", "rafaga_max_kmh", "temp_max_c"
    ]).to_dicts()

    # Format data for Chart.js
    dist_riesgo = resumen["distribucion_riesgo"]
    labels_chart = ["Extremo", "Alto", "Moderado", "Bajo", "Nulo"]
    data_chart = [
        dist_riesgo.get("EXTREMO", 0),
        dist_riesgo.get("ALTO", 0),
        dist_riesgo.get("MODERADO", 0),
        dist_riesgo.get("BAJO", 0),
        dist_riesgo.get("NULO", 0)
    ]

    deptos_chart_labels = [d["departamento"] for d in resumen["deptos_criticos"][:8]]
    deptos_chart_data = [d["km2_extremo"] + d["km2_alto"] + d["km2_moderado"] for d in resumen["deptos_criticos"][:8]]

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SAT-Incendios PY - Tablero Operativo Diario</title>
  <!-- Leaflet CSS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <!-- Google Fonts & Bootstrap Icons -->
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
  <!-- Chart.js -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', sans-serif;
      background-color: #0f172a;
      color: #f8fafc;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }}
    header {{
      background: #1e293b;
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #334155;
      z-index: 1000;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .brand i {{ font-size: 24px; color: #ef4444; }}
    .brand h1 {{ font-size: 18px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; }}
    .brand span {{ font-size: 13px; color: #94a3b8; margin-left: 8px; }}
    .kpi-bar {{
      display: flex;
      gap: 16px;
    }}
    .kpi-pill {{
      background: #334155;
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 13px;
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .kpi-pill strong {{ color: #ffffff; }}
    .main-container {{
      display: flex;
      flex: 1;
      height: calc(100vh - 60px);
      position: relative;
    }}
    #sidebar {{
      width: 420px;
      background: #1e293b;
      border-right: 1px solid #334155;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      z-index: 500;
    }}
    .panel-section {{
      padding: 18px 20px;
      border-bottom: 1px solid #334155;
    }}
    .section-title {{
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: #94a3b8;
      font-weight: 700;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    #map {{
      flex: 1;
      height: 100%;
      background: #0b1120;
    }}
    .depto-row {{
      display: flex;
      justify-content: space-between;
      padding: 8px 10px;
      background: #0f172a;
      border-radius: 6px;
      margin-bottom: 6px;
      font-size: 13px;
      align-items: center;
    }}
    .badge-risk {{
      padding: 3px 8px;
      border-radius: 4px;
      font-weight: 600;
      font-size: 11px;
    }}
    .badge-extremo {{ background: #991b1b; color: #fecaca; }}
    .badge-alto {{ background: #c2410c; color: #ffedd5; }}
    .badge-moderado {{ background: #854d0e; color: #fef08a; }}
    .badge-bajo {{ background: #334155; color: #cbd5e1; }}
    .hotspot-card {{
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 10px 14px;
      margin-bottom: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .hotspot-card:hover {{
      border-color: #ef4444;
      background: #1e293b;
    }}
    .hotspot-header {{
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 6px;
    }}
    .hotspot-metrics {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      font-size: 11px;
      color: #94a3b8;
    }}
    .leaflet-popup-content-wrapper {{
      background: #1e293b;
      color: #f8fafc;
      border-radius: 8px;
      padding: 8px;
      font-family: 'Inter', sans-serif;
    }}
    .leaflet-popup-tip {{ background: #1e293b; }}
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <i class="bi bi-fire"></i>
      <h1>SAT-Incendios PY</h1>
      <span>Sistema de Alerta Temprana Biofísico GeoAI</span>
    </div>
    <div class="kpi-bar">
      <div class="kpi-pill"><i class="bi bi-calendar3"></i> Fecha: <strong>{fecha}</strong></div>
      <div class="kpi-pill"><i class="bi bi-grid-3x3"></i> Celdas: <strong>400,006 km²</strong></div>
      <div class="kpi-pill"><i class="bi bi-exclamation-triangle-fill" style="color: #ef4444;"></i> Alerta Roja: <strong>{dist_riesgo.get('EXTREMO', 0):,} km²</strong></div>
      <div class="kpi-pill"><i class="bi bi-shield-fill-check" style="color: #22c55e;"></i> Normal: <strong>{dist_riesgo.get('BAJO', 0):,} km²</strong></div>
    </div>
  </header>

  <div class="main-container">
    <div id="sidebar">
      
      <!-- Distribución de Riesgo -->
      <div class="panel-section">
        <div class="section-title"><i class="bi bi-pie-chart-fill"></i> Distribución de Peligro Nacional</div>
        <canvas id="chartRiesgo" height="150"></canvas>
      </div>

      <!-- Departamentos Críticos -->
      <div class="panel-section">
        <div class="section-title"><i class="bi bi-building"></i> Departamentos en Monitoreo</div>
        <div id="lista-deptos"></div>
      </div>

      <!-- Top Focos Críticos -->
      <div class="panel-section">
        <div class="section-title"><i class="bi bi-crosshair"></i> Top Focos de Máxima Probabilidad</div>
        <div id="lista-hotspots"></div>
      </div>

    </div>

    <!-- Mapa Leaflet -->
    <div id="map"></div>
  </div>

  <!-- Leaflet JS -->
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <script>
    // 1. Inicializar Mapa Centrado en Paraguay
    const map = L.map('map', {{
      center: [-23.4425, -58.4438],
      zoom: 7,
      zoomControl: true
    }});

    // Capa Base CartoDB DarkMatter
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; CartoDB &copy; OpenStreetMap',
      subdomains: 'abcd',
      maxZoom: 19
    }}).addTo(map);

    // Datos GeoJSON de Departamentos
    const deptosGeojson = {json.dumps(deptos_geojson)};
    L.geoJSON(deptosGeojson, {{
      style: {{
        color: '#64748b',
        weight: 1.2,
        fillColor: '#334155',
        fillOpacity: 0.15
      }},
      onEachFeature: function (feature, layer) {{
        layer.bindTooltip(feature.properties.shapeName || feature.properties.departamento, {{
          permanent: false,
          direction: 'center',
          className: 'depto-label'
        }});
      }}
    }}).addTo(map);

    // 2. Colores por Nivel de Riesgo
    const colores = {{
      'EXTREMO': '#ef4444',
      'ALTO': '#f97316',
      'MODERADO': '#eab308',
      'BAJO': '#64748b',
      'NULO': '#334155'
    }};

    // 3. Renderizar Puntos Críticos en el Mapa
    const puntos = {json.dumps(puntos_mapa)};
    const markers = [];

    puntos.forEach(p => {{
      const color = colores[p.nivel_alerta] || '#64748b';
      const circle = L.circleMarker([p.lat, p.lon], {{
        radius: p.nivel_alerta === 'EXTREMO' ? 6 : (p.nivel_alerta === 'ALTO' ? 5 : 3.5),
        fillColor: color,
        color: '#ffffff',
        weight: 0.5,
        opacity: 0.9,
        fillOpacity: 0.75
      }}).addTo(map);

      circle.bindPopup(`
        <div style="font-size: 13px;">
          <h4 style="margin: 0 0 6px 0; font-size: 14px; color: ${{color}};">Alerta ${{p.nivel_alerta}}</h4>
          <p style="margin: 2px 0;"><strong>Celda ID:</strong> ${{p.cell_id}}</p>
          <p style="margin: 2px 0;"><strong>Depto:</strong> ${{p.departamento}} (${{p.region}})</p>
          <p style="margin: 2px 0;"><strong>P(Fuego):</strong> <b>${{(p.probabilidad_fuego * 100).toFixed(1)}}%</b></p>
          <hr style="border: 0; border-top: 1px solid #334155; margin: 6px 0;">
          <p style="margin: 2px 0;"><strong>FWI ISI:</strong> ${{p.isi.toFixed(1)}}</p>
          <p style="margin: 2px 0;"><strong>VPD:</strong> ${{p.vpd_max_kpa.toFixed(2)}} kPa</p>
          <p style="margin: 2px 0;"><strong>Ráfaga:</strong> ${{p.rafaga_max_kmh.toFixed(1)}} km/h</p>
        </div>
      `);
      markers.push({{ id: p.cell_id, marker: circle, lat: p.lat, lon: p.lon }});
    }});

    // 4. Renderizar Gráfico Donut de Riesgo
    const ctx = document.getElementById('chartRiesgo').getContext('2d');
    new Chart(ctx, {{
      type: 'doughnut',
      data: {{
        labels: {json.dumps(labels_chart)},
        datasets: [{{
          data: {json.dumps(data_chart)},
          backgroundColor: ['#ef4444', '#f97316', '#eab308', '#64748b', '#334155'],
          borderWidth: 0
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', boxWidth: 12 }} }}
        }}
      }}
    }});

    // 5. Llenar Lista de Departamentos
    const deptosData = {json.dumps(resumen["deptos_criticos"])};
    const listaDeptos = document.getElementById('lista-deptos');
    deptosData.slice(0, 6).forEach(d => {{
      const totalCrit = d.km2_extremo + d.km2_alto + d.km2_moderado;
      const row = document.createElement('div');
      row.className = 'depto-row';
      row.innerHTML = `
        <div>
          <strong>${{d.departamento}}</strong>
          <div style="font-size: 11px; color: #94a3b8;">${{d.region}}</div>
        </div>
        <div style="text-align: right;">
          <span class="badge-risk ${{d.km2_extremo > 0 ? 'badge-extremo' : (d.km2_alto > 0 ? 'badge-alto' : 'badge-moderado')}}">
            ${{totalCrit.toLocaleString()}} km²
          </span>
          <div style="font-size: 11px; color: #94a3b8;">P: ${{d.prob_media.toFixed(3)}}</div>
        </div>
      `;
      listaDeptos.appendChild(row);
    }});

    // 6. Llenar Lista de Top Hotspots
    const hotspotsData = {json.dumps(resumen["top_10_celdas_criticas"])};
    const listaHotspots = document.getElementById('lista-hotspots');
    hotspotsData.forEach(h => {{
      const card = document.createElement('div');
      card.className = 'hotspot-card';
      card.innerHTML = `
        <div class="hotspot-header">
          <span>${{h.departamento}}</span>
          <span style="color: #ef4444;">${{(h.probabilidad_fuego * 100).toFixed(1)}}%</span>
        </div>
        <div class="hotspot-metrics">
          <div>ISI: <b>${{h.isi.toFixed(1)}}</b></div>
          <div>VPD: <b>${{h.vpd_max_kpa.toFixed(1)}} kPa</b></div>
          <div>Viento: <b>${{h.viento_max_kmh.toFixed(0)}} km/h</b></div>
        </div>
      `;
      card.onclick = () => {{
        map.flyTo([h.lat, h.lon], 12);
        const target = markers.find(m => m.id === h.cell_id);
        if (target) target.marker.openPopup();
      }};
      listaHotspots.appendChild(card);
    }});
  </script>
</body>
</html>
"""

    dir_out = ROOT_DIR / "entregables"
    dir_out.mkdir(parents=True, exist_ok=True)
    p_html = dir_out / "dashboard_alerta_interactivo.html"
    p_html.write_text(html_content, encoding="utf-8")

    tamanio_kb = p_html.stat().st_size / 1024
    print(f"\n✓ Dashboard HTML generado con éxito: {p_html.name} ({tamanio_kb:.1f} KB)")
    print(f"✓ Ruta absoluta: {p_html}")
    print("=" * 65)
    return p_html

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera dashboard interactivo HTML para SAT-Incendios PY.")
    parser.add_argument("--fecha", type=str, default=None, help="Fecha en formato YYYY-MM-DD")
    args = parser.parse_args()
    generar_dashboard_interactivo(args.fecha)
