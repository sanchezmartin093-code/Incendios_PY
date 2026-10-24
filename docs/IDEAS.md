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
