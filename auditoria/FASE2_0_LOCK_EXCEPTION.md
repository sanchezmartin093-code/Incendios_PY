# Fase 2.0 - Excepción formal: hashes del lock

**Fecha:** 2026-10-27  
**Estado:** excepción aceptada para cerrar Fase 2.0.

## Qué se intentó

1. `pip-compile --generate-hashes` sobre `requirements.txt`.
2. `pip-compile --generate-hashes --no-emit-index-url --output-file requirements-lock-hashed.txt requirements.txt`.

Resultado:

- El primer intento superó 600 s.
- El segundo intento superó 1.200 s.
- No se generó `requirements-lock-hashed.txt`.
- No hubo error de código; el proceso quedó esperando resolución/red.

## Estado actual

- `requirements-lock.txt` contiene **versiones exactas** para las 54 dependencias.
- El CI instala ese lock y ejecuta `pip check` y `pytest`.
- El lock **no** incluye hashes de ruedas y no se usa `--require-hashes`.

## Decisión

Se acepta la excepción para cerrar Fase 2.0:

- No se inventaron hashes.
- No se afirma reproducibilidad criptográfica de las ruedas.
- La reproducibilidad del dataset sí está verificada por SHA-256 (8/8 particiones idénticas).
- La excepción debe revisarse antes de un release público o de Fase 3.
- El CI mantiene control de versiones exactas, `pip check` y tests.

## Consecuencia

Fase 2.0 queda cerrada con una limitación explícita y documentada, no con un pendiente ambiguo.
