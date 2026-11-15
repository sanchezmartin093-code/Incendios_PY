# Protocolo de Desidentificación de Datos Clínicos y Epidemiológicos

> **Estándar:** HIPAA Safe Harbor Method (45 CFR § 164.514(b)) + Criterio de Riesgo Estadístico de Re-identificación

Este documento formaliza los procedimientos de anonimización aplicados a cualquier microdato primario antes de su almacenamiento en `datos/raw/` o su depósito público en Zenodo.

---

## 1. Supresión de los 18 Identificadores Directos

Queda estrictamente prohibida la inclusión en el repositorio de los siguientes elementos:

1. Nombres y apellidos de participantes, pacientes o víctimas.
2. Subdivisiones geográficas menores a nivel departamental (direcciones exactas, números de casa, georreferenciaciones puntuales con precisión submunicipal).
3. Fechas exactas de nacimiento o admisión (se transforman a rangos quinquenales o edad cumplida).
4. Números de teléfono o fax.
5. Direcciones de correo electrónico.
6. Números de Cédula de Identidad (C.I.) o pasaporte.
7. Números de historia clínica o expediente médico.
8. Números de beneficiario de seguros o planes de salud.
9. Números de cuenta o matrícula profesional.
10. Números de certificado o licencia.
11. Identificadores de vehículos (placas, números de motor o chasis).
12. Identificadores de dispositivos o números de serie.
13. URLs de perfiles o páginas personales.
14. Direcciones IP.
15. Identificadores biométricos (huellas, voz, retina).
16. Fotografías de rostro completo o imágenes identificables.
17. Cualquier otro número, código o característica identificadora única.
18. Datos de participantes con edad igual o superior a 90 años (se agrupan en la categoría única `≥90`).

---

## 2. Manejo de Datos Primarios Identificables (`_privado/`)

- Cualquier base de datos cruda original que contenga identificadores directos se almacena temporalmente y de forma aislada en la carpeta `_privado/`.
- La carpeta `_privado/` reside permanentemente en `.gitignore` y nunca se sube a GitHub ni se incluye en respaldos abiertos.
- El script de desidentificación debe ejecutarse localmente, transformando los datos crudos hacia `datos/raw/` en formato estrictamente anónimo.
