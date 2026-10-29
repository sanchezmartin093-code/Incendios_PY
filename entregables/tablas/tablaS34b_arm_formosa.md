| Arm / variant | 2024 M2 / M3 ROC | 2025 M2 / M3 ROC | Note |
|---|---:|---:|---|
| G0-Z | 0.7133 / 0.7245 | 0.7050 / 0.7112 | frozen |
| G0-C | 0.7133 / 0.7245 | 0.7050 / 0.7112 | local recalibration |
| G0-R | 0.7178 / 0.7275 | 0.7144 / 0.7150 | PY+Formosa retrain |
| H-B (class 6 → wetland) | 0.7170 / 0.7256 | 0.7066 / 0.7127 | sensitivity |
| H-C (mask class 6 ≥50 %) | 0.7508 / 0.7616 | 0.7324 / 0.7357 | removes 22 % area, 43 % detections, 33 % positives |
| t−1 buffer 20 km | 0.7058 / 0.7173 | 0.6967 / 0.7041 | G0-C |
| t−1 buffer 30 km | 0.7148 / 0.7251 | 0.7014 / 0.7111 | G0-C |
