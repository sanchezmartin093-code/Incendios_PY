# Banco de Ideas, Futuras Extensiones y Fundamentación Teórica
## Proyecto `Incendios_PY`

---

## 💡 1. Benchmarking Retrospectivo: Comparación con Modelos Estándares Avalados

> **Objetivo futuro:** Una vez entrenado y calibrado el modelo de `Incendios_PY`, contrastar cuantitativamente su desempeño predictivo frente a los modelos normativos oficialmente avalados y utilizados por las agencias del Estado.

### Modelos de Referencia para el Benchmarking:
1. **Canadian Forest Fire Weather Index (FWI) puro:**
   * Utilizado a nivel internacional por Copernicus EFFIS y adoptado como estándar por agencias forestales.
2. **Índice de "Risco de Fogo" del INPE (Brasil):**
   * El modelo semi-empírico de referencia regional en Sudamérica que alimenta las alertas actuales de INFONA y MADES en Paraguay.
3. **Línea Base Climatológica / Persistencia:**
   * Probabilidad histórica media acumulada por celda.

### Métricas de Contraste a Reportar en la Discusión del Paper:
* **Brier Skill Score (BSS):** Demostrar la ganancia neta de calibración de nuestro modelo frente al FWI y Risco de Fogo.
* **Curvas PR-AUC (Precision-Recall):** Comparar la capacidad de aislar igniciones verdaderas sin inflar el área de alerta.
* **Tasa de Reducción de Falsas Alarmas (Especificidad Operativa):** Cuántos falsos avisos se ahorran los cuarteles de bomberos voluntarios (CBVP) y brigadas de parques nacionales.

---

## 🏛️ 2. Fundamentación Teórica e Institucional (Para la Sección de Discusión)

### A. La Inercia Burocrática y el "Blindaje Legal"
* **Aversión al riesgo regulatorio:** Los organismos del Estado y los tomadores de decisiones tienden a operar con modelos antiguos (como el FWI de 1987 o índices estáticos de los años 90) no porque sean más precisos, sino porque constituyen un **estándar normativo legalmente blindado**.
* Si un funcionario público activa protocolos de emergencia o despacha brigadas apoyándose en un índice oficial estandarizado, su decisión está administrativamente respaldada ante auditorías y comisiones. Adoptar algoritmos de Machine Learning exige vencer una barrera de desconfianza institucional y de responsabilidad legal ante eventuales errores.

### B. La Brecha Tecnológica y la Maduración Reciente (2021–2024)
* Este modelo es viable hoy porque convergen avances de muy reciente adopción pública:
  * **MapBiomas Cono Sur (Chaco y Bosque Atlántico):** Cartografía anual dinámica a 30m que refleja los cambios vertiginosos de uso de suelo en Paraguay.
  * **Democratización de Reanálisis Climático:** Disponibilidad abierta e instantánea de datos horarios de ERA5-Land.
  * **Motores Zero-Copy (DuckDB / Apache Arrow):** Capacidad de procesar mallas masivas espaciotemporales de cientos de miles de celdas en hardware convencional sin requerir infraestructura cloud millonaria.

### C. Superación del Cuello de Botella Operativo
* **Propósito del proyecto:** Desarrollar este sistema bajo los principios de **código abierto, reproducibilidad estricta y rigor de publicación científica** demuestra empíricamente cómo la tecnología contemporánea puede superar la dependencia de reportes estáticos diarios en PDF (que llegan con 12 a 24 horas de retraso) y transformar la gestión de desastres en Paraguay de una postura puramente reactiva a una preventiva y anticipatoria.

---

## 🛰️ 3. Integración Operativa de Verificación Satelital Multiespectral Tri-Modal

> **Idea para incorporación al pipeline:** Cerrar el ciclo entre el pronóstico predictivo ($t+24\text{h}$) y la confirmación en tiempo real ($t+0\text{h}$ a $t+3\text{h}$) integrando un bucle de verificación satelital tri-modal multiescala para cada nodo o celda en Alerta Roja ($P > 93\%$).

### Las 3 Capas de Verificación Satelital:
1. **Nivel 1: Detección Continua Sub-Horaria (GOES-16 ABI - Cada 10 a 15 min):**
   * *Función:* Monitoreo dinámico del despegue de la pluma de humo (GeoColor) y de la anomalía infrarroja de 3.9 µm en tiempo real.
   * *Utilidad:* Permite a la SEN y cuerpos de bomberos validar si el nodo alertado a las 06:00 AM inició efectivamente actividad convectiva durante la siesta, alertando hacia dónde se desplazará el humo ($PM_{2.5}$).
2. **Nivel 2: Detección Térmica de Alta Precisión (VIIRS 375m Suomi-NPP / NOAA-20 / NOAA-21):**
   * *Función:* Descarga automatizada vía API de NASA LANCE / FIRMS de los pases orbitales (13:30h y 01:30h).
   * *Utilidad:* Superposición de píxeles térmicos activos (banda I4 de 3.75 µm y potencia radiativa FRP) sobre la grilla UTM de 1 km² para confirmar avance del frente de fuego y auditar la tasa de acierto del modelo en tiempo casi real.
3. **Nivel 3: Penetración Óptica de Humo en Infrarrojo de Onda Corta (SWIR / Sentinel-2 & VIIRS M11-I2-I1):**
   * *Función:* Descomposición radiativa en falso color SWIR (1.6 a 2.2 µm). A estas longitudes de onda, las partículas de humo se vuelven prácticamente transparentes (*smoke penetration*).
   * *Utilidad:* Aísla la línea de llama activa (resplandor incandescente naranja/rojo) frente a la cicatriz de quema fría (marrón oscuro/negro) y la vegetación sana intacta (verde brillante). Fundamental para cuantificar hectáreas netas quemadas y coordinar ataque aéreo en zonas donde el humo ciega a los pilotos de hidrantes.

### Arquitectura de Implementación Propuesta:
* Script automatizado `scripts/14_verificacion_satelital_alerta.py` que:
  1. Lea el boletín diario y extraiga las celdas con `probabilidad_fuego >= 0.80` (Alerta Roja).
  2. Consulte las APIs WMS/WMTS de NASA GIBS y Copernicus Dataspace.
  3. Genere automáticamente el tríptico de verificación satelital (`fig06_comparativa_satelital_tres_modos.png`) y lo adjunte como anexo visual al boletín técnico diario para las autoridades.
