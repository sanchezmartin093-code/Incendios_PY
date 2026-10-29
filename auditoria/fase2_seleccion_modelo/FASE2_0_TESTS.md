# FASE 2.0 - Verificación final

fecha_utc: 2026-10-27
comando: /tmp/auditvenv/bin/python -m pytest -q tests
resultado: 71 passed, 2 warnings

comando: /tmp/auditvenv/bin/python scripts/00_validar_datos.py
resultado: 0 FAIL

dataset: 8/8 particiones identicas entre dos builds
holdout_confirmacion: 31 bloques, 68719 celdas, 95031 celdas buffer
train_min_fecha: 2018-03-02
fuel_futuro_train_val_test: 0
