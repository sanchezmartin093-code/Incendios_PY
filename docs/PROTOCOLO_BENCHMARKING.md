---
title: "Protocolo de Benchmarking Internacional para Validación de Modelos Predictivos de Incendios Forestales"
date: 2026-10-24
tags:
  - ciencia-del-fuego
  - geoai
  - teledeteccion
  - machine-learning
  - paraguay
  - evaluacion-modelos
---

# Protocolo de Benchmarking Internacional para Validación de Modelos Predictivos de Incendios Forestales (GeoAI vs. Estándares Globales)

> **Contexto:** Guía metodológica para auditar y comparar objetivamente modelos de peligro y probabilidad de incendios forestales (p. ej., `Incendios_PY`) frente a sistemas operativos canónicos internacionales (**Canadian FWI / Copernicus EFFIS**, **INPE Risco de Fogo**, **USFS NFDRS**) bajo los estándares de revistas Q1 (*Remote Sensing of Environment*, *Nature Communications*).

---

## ⚠️ 1. La Trampa Metodológica: El "Data Leakage" en la Literatura

En machine learning aplicado a ciencias de la Tierra y ecología del fuego, existe una brecha crítica entre papers académicos y sistemas operativos en producción:

1. **La Falsa Precisión del 95%+:** 
   Ocurre cuando se aplica validación cruzada aleatoria clásica (*Random K-Fold*). Dado que los incendios tienen fuerte autocorrelación espacial (una pluma de humo abarca decenas de kilómetros), una celda de prueba contigua a una de entrenamiento permite que el árbol de decisión "copie" el estado del vecino. Esto infla artificialmente el ROC-AUC a $0.90\text{--}0.98$, pero el modelo colapsa en el mundo real.
2. **El Protocolo de Rigor Falsable:**
   * **Partición Out-of-Time:** Entrenar con años pasados (ej. 2018–2022) y evaluar a ciegas en años futuros no vistos (ej. 2024).
   * **Dead-Zone Spatial Buffer (20 km):** Las pseudo-ausencias y muestras de validación deben distar al menos 20 km de cualquier foco de presencia activa (Roberts et al., 2017; Ploton et al., 2020), anulando la autocorrelación del semivariograma.

Bajo este rigor, los modelos operativos internacionales reales rinden entre **ROC-AUC 0.65 y 0.78**.

---

## 🏛️ 2. Modelos Contendientes de Referencia

Para una comparativa formal en Sudamérica, se deben contrastar cuatro arquitecturas sobre el **mismo conjunto de prueba ciego**:

| Contendiente | Origen / Ámbito | Tipo de Modelo | Variables Principales |
| :--- | :--- | :--- | :--- |
| **1. Incendios_PY** | Paraguay (Local) | Machine Learning discriminante (LightGBM C++) + Two-Stage Bio-physical Gating + Calibración de Platt Monótona. | 22 variables: ERA5-Land, FWI parciales, MapBiomas 30m, amplitud térmica, $VPD \times CDD$, viento$^2$. |
| **2. Canadian FWI Puro** | Canadá / Adoptado por Copernicus EFFIS y GWIS | Empírico de balance hídrico foliar para coníferas (Van Wagner, 1987). | FFMC, DMC, DC, ISI, BUI, FWI (requiere mediodía solar). |
| **3. INPE Risco de Fogo** | Brasil / Adoptado por INFONA y MADES en Paraguay | Semi-empírico exponencial calibrado para Cerrado/Amazonía (Setzer et al., 2019). | Días secos consecutivos ($CDD$), humedad relativa mínima, temperatura máxima y clases de vegetación macro. |
| **4. Línea Base Climatológica** | Referencia estadística nula | Frecuencia histórica acumulada de fuego en la celda o ecorregión ($P_{\text{base}} = \bar{Y}_{\text{train}}$). | Probabilidad marginal constante. |

---

## 📐 3. Métricas de Evaluación para Eventos Raros

En problemas donde la prevalencia de incendios es $< 10\%$, la Exactitud Global (*Accuracy*) es engañosa. El benchmarking debe reportar:

### A. Área bajo la Curva Precision-Recall (PR-AUC)
* A diferencia del ROC-AUC (que puede ser optimista porque la tasa de verdaderos negativos es enorme), el PR-AUC ignora los verdaderos negativos y se concentra exclusivamente en la capacidad del modelo de aislar igniciones verdaderas sin generar falsas alarmas masivas.
* La línea base aleatoria en PR-AUC es igual a la prevalencia empírica $\pi$:
  $$\text{PR-AUC}_{\text{azar}} = \pi = \frac{N_{\text{fuegos}}}{N_{\text{total}}}$$

### B. Brier Skill Score (BSS)
Mide la ganancia porcentual en calibración probabilística respecto a la referencia climatológica:
$$\text{Brier Score (BS)} = \frac{1}{N} \sum_{i=1}^N (P_i - Y_i)^2$$
$$\text{BSS} = 1 - \frac{\text{BS}_{\text{modelo}}}{\text{BS}_{\text{climatología}}}$$
* $\text{BSS} > 0$: El modelo aporta valor predictivo real superior a la climatología histórica.
* $\text{BSS} \le 0$: El modelo es peor o igual que usar el promedio histórico.

### C. Métricas en Punto de Operación con Umbral $F_{0.5}$
Para cuerpos de bomberos y emergencias, un falso positivo (movilización inútil de recursos escasos) es costoso. La métrica $F_{0.5}$ pondera la **Precisión el doble que la Sensibilidad (Recall)**:
$$F_{0.5} = (1 + 0.5^2) \frac{\text{Precisión} \times \text{Recall}}{0.5^2 \cdot \text{Precisión} + \text{Recall}} = 1.25 \frac{\text{Precisión} \times \text{Recall}}{0.25 \cdot \text{Precisión} + \text{Recall}}$$

---

## 📊 4. Análisis de Utilidad Operativa (Cost-Loss Analysis)

El valor económico relativo $V$ de un sistema de alerta temprana se calcula en función de la matriz de contingencia:
* $C$: Costo de tomar una acción preventiva (ej. despacho de patrulla, prohibición de quema).
* $L$: Pérdida si ocurre un incendio sin acción previa ($L > C$).
* Ratio de costo/pérdida: $\alpha = C / L \in (0, 1)$.

$$V(\alpha) = \frac{\min(\alpha, \pi) - [\alpha \cdot (\text{TP} + \text{FP}) + \text{FN}]}{\min(\alpha, \pi) - \alpha \cdot \pi}$$

Un modelo superior debe mantener $V(\alpha) > 0$ en el rango operativo típico de protección civil ($\alpha \in [0.05, 0.25]$).

---

## 📈 5. Resultados Empíricos del Benchmarking (Evaluación Out-of-Time 2024)

Evaluación cuantitativa sobre el conjunto de testeo ciego con Dead-Zone Buffer de 20 km, desacoplando **Ignición Primaria (Física Pura)** y **Propagación Táctica (Contagio $t-1$)**:

| Región | Modelo Predictivo | ROC-AUC | PR-AUC | Precisión ($F_{0.5}$) | Sensibilidad | Falsos Positivos | Brier Skill Score | $F_{0.5}$-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oriental** | **Incendios_PY (Ignición Física)** | **0.7711** | **0.2810** | **34.09%** | 37.50% | **29** | **+0.0740** | **0.3472** |
| **Oriental** | **Incendios_PY (Propagación Táctica)**| **0.8650** | **0.5430** | **100.00%**| 22.50% | **0** | **+0.2910** | **0.5921** |
| Oriental | Proxy INPE Risco de Fogo | 0.3995 | 0.0703 | 9.69% | 85.00% | 317 | -8.2341 | 0.1177 |
| Oriental | Canadian FWI (ISI) | 0.3892 | 0.0779 | 10.25% | 92.50% | 324 | -2.7331 | 0.1247 |
| Oriental | Línea Base Climatológica | 0.5000 | 0.0909 | 0.00% | 0.00% | **0** | +0.0000 | 0.0000 |
| **Chaco** | **Incendios_PY (Ignición Física)** | **0.5952** | **0.1290** | **23.88%** | 19.28% | **102** | **+0.0013** | **0.2279** |
| **Chaco** | **Incendios_PY (Propagación Táctica)**| **0.9410** | **0.8486** | **97.71%** | 77.11% | **3** | **+0.6664** | **0.9275** |
| Chaco | Proxy INPE Risco de Fogo | 0.6457 | 0.1314 | 13.89% | 72.89% | 750 | -8.0707 | 0.1658 |
| Chaco | Canadian FWI (ISI) | 0.5432 | 0.0967 | 11.25% | 78.31% | 1026 | -0.9321 | 0.1357 |
| Chaco | Línea Base Climatológica | 0.5000 | 0.0909 | 0.00% | 0.00% | **0** | +0.0000 | 0.0000 |

### Conclusiones de Rendimiento y Honestidad Metodológica:
1. **Desacoplamiento Ignición vs. Propagación:**
   - La capa de **Ignición Primaria** predice el nacimiento espontáneo o antrópico de fuegos nuevos sin utilizar atajos espaciales ni información de focos previos en $t-1$. En la Región Oriental supera con creces al FWI y al INPE (ROC 0.7711 vs. sub-azar <0.40) y reduce las falsas alarmas en un **91.0%** (29 FP frente a 324 del FWI).
   - La capa de **Propagación Táctica** modela el avance condicional de frentes de fuego activos ya detectados por satélite ($t-1$), alcanzando un ROC de **0.9410** en el Chaco con una precisión del **97.71%** y solo 3 falsas alarmas sobre 1.660 celdas no quemadas.
2. **Mitigación Masiva del Desgaste de Cuadrillas:**
   - Frente a los 1.026 falsos avisos del FWI canadiense y 750 del INPE en el Chaco, la capa de Ignición Física reduce los falsos positivos a 102, y la capa de Propagación los reduce a solo 3, permitiendo una asignación quirúrgica de aeronaves y camiones cisterna de la SEN y el CBVP.
3. **Calibración Probabilística Real (BSS):**
   - Incendios_PY es el único modelo que obtiene Brier Skill Score positivo en ambas regiones y capas (+0.0740 / +0.2910 en Oriental, +0.0013 / +0.6664 en Chaco), mientras que el FWI y el INPE exhiben descalibraciones severas ($BSS < -0.90$).

---

## 📚 6. Referencias Normativas

- **Ploton, P., et al. (2020).** Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, 11(1), 4540.
- **Roberts, D. R., et al. (2017).** Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure. *Ecography*, 40(8), 913–929.
- **Setzer, A. W., et al. (2019).** *O produto "Risco de Fogo" do INPE: Metodología e calibração da versão 10*. Instituto Nacional de Pesquisas Espaciais (INPE).
- **Van Wagner, C. E. (1987).** *Development and structure of the Canadian Forest Fire Weather Index System*. Canadian Forestry Service.
