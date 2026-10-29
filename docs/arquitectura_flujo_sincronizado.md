# Arquitectura del Flujo Sincronizado: Google Cloud & Antigravity

**Fecha de Actualización:** 28 de Octubre de 2026  
**Autor:** Martín Sánchez Insfrán  
**Repositorio Base:** `Incendios_PY` (v2.1.0) & Ecosistema de Ciencia Cuantitativa en Paraguay (`paraguay-en-datos`, `Homicidios_PY`, `Calor_PY`).

---

## 1. El Esquema General del Pipeline

```mermaid
flowchart TD
    classDef cloudBox fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#ffffff;
    classDef localBox fill:#1e293b,stroke:#4ade80,stroke-width:2px,color:#ffffff;
    classDef ideBox fill:#1e293b,stroke:#f59e0b,stroke-width:2px,color:#ffffff;
    classDef connector fill:#0f172a,stroke:#94a3b8,stroke-dasharray: 4 4,color:#e2e8f0;

    subgraph CLOUD["1. EN LA NUBE (Google Cloud - Subvenciones Startups & TRC)"]
        direction TB
        C1["Vertex AI (Gemini 3.8 Flash Multimodal)<br/>Auditoría satelital tri-modal (GOES-16, VIIRS, SWIR)"]
        C2["Google TPU Research Cloud (TPUs v2/v3/v4)<br/>Cómputo tensorial masivo y Conformal Prediction"]
        C3["Google Cloud Storage / Cloud Run<br/>Almacenamiento Parquet y API de alertas en tiempo real"]
    end
    class CLOUD cloudBox;

    subgraph LOCAL["2. EN DISCO LOCAL (Proyectos_PY / Incendios_PY)"]
        direction TB
        L1["DuckDB & Polars (Zero-Copy Engine)<br/>Procesamiento de 397.448 celdas y 915M celda-días"]
        L2["Archivos ligeros de salida<br/>resumen_fuego_hoy.json (~5 KB) / matrices Parquet"]
    end
    class LOCAL localBox;

    subgraph IDE["3. EN ANTIGRAVITY (IDE Agéntico / Pair-Programming)"]
        direction TB
        A1["Lectura ultraligera de métricas (~200 tokens)<br/>Sin saturar la ventana de contexto"]
        A2["Generación de figuras vectoriales de alta resolución"]
        A3["Redacción científica y compilación Quarto (.qmd -> PDF/HTML)"]
    end
    class IDE ideBox;

    CLOUD -->|Descarga de matrices y reportes JSON| LOCAL
    LOCAL -->|Lectura local de baja latencia| IDE
```

---

## 2. Roles Claros de Cada Herramienta

| Componente | Qué hace | Consumo de Tokens / Costo |
| :--- | :--- | :--- |
| **Google Cloud (Vertex AI + TPUs vía TRC)** | Trabajo pesado: Ingesta satelital diaria, auditoría visual de plumas de humo e incandescencia en falso color SWIR, entrenamiento tensorial masivo. | Se absorbe con los **$2.000 USD de Google for Startups** y el programa gratuito de **Google TRC**. |
| **Tu Computadora (Mac)** | Ejecuta scripts locales en Python (v2.1.0), gestiona la base analítica en DuckDB/Polars y compila manuscritos en Quarto (`quarto render`). | **0 tokens.** Cómputo local nativo de alto rendimiento. |
| **Antigravity (Chat / Agente)** | Arquitectura de código, refactorización, redacción científica por secciones y control de calidad metodológico para *Computers & Geosciences*. | **Cuota de Antigravity optimizada** (usando Gemini 3.8 Flash para máxima eficiencia). |

---

## 3. Guía de Buenas Prácticas para no Saturar Antigravity

1. **Trabajar por Secciones:** Evitar pedir la lectura de datasets completos o manuscritos de más de 50 KB en una sola llamada. Enfocar los prompts en secciones puntuales (ej. *Sección 3.2: Decoupled Target Formulation*).
2. **Uso de Archivos Intermedios:** Hacer que los scripts pesados de Python/DuckDB guarden resúmenes estadísticos condensados en JSON o Parquet ligero; Antigravity solo necesita leer ese resumen para redactar los párrafos científicos.
3. **Selección del Modelo:** Mantener seleccionado **Gemini 3.8 Flash** en la interfaz para el trabajo ágil de código y análisis; reservar modelos Pro para revisiones teóricas específicas.
4. **Mantenimiento del Contexto:** Ejecutar el comando `/compact` periódicamente cuando el hilo de conversación acumule muchas iteraciones.
