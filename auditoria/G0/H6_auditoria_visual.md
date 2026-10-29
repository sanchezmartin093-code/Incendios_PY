# H6 — Auditoría visual (piloto de 60 puntos)

## Diseño

Se generó una muestra estratificada de **300 celdas**:

- 60 bosque no inundable;
- 60 clase 6 (bosque inundable);
- 60 humedal;
- 60 agricultura;
- 60 aleatorias.

Para el piloto se revisaron visualmente **12 por estrato (60 puntos)** con imágenes ESRI World Imagery centradas en cada celda. La revisión fue de consistencia gruesa, no de polígono: bosque/leñosa, humedal/inundado, cultivo/pastura, o no verificable.

## Resultado del piloto

| Estrato | Revisados | Consistentes | Acuerdo |
|---|---:|---:|---:|
| Bosque no inundable | 12 | 12 | 100 % |
| Clase 6 (bosque inundable) | 12 | 11 | 91.7 % |
| Humedal | 12 | 11 | 91.7 % |
| Agricultura | 12 | 8 | 66.7 % |
| Aleatorio | 12 | 11 | 91.7 % |
| **Total** | **60** | **53** | **88.3 %** |

Errores observados:

- 1 clase 6 con imagen de bosque cerrado sin evidencia de inundación.
- 1 humedal con imagen de campo seco.
- 4 agriculturas que parecen pastura o bosque, no cultivo anual.
- 1 punto aleatorio sin datos/mezcla.

## Interpretación

- Bosque, clase 6 y humedal muestran consistencia razonable en el piloto.
- **Agricultura es el estrato más débil**: varias celdas clasificadas como cultivo anual parecen pastura o bosque. Esto importa porque `pct_agricultura_anual` y `pct_pastura_ganadera` alimentan M2/M3.
- El piloto **no reemplaza** los 300 puntos del protocolo. La muestra completa está en `H6_muestra_300.csv` y las imágenes del piloto en `h6_chips/`.
- No se puede declarar H6 cerrado con 60/300; sí se puede afirmar que no aparecen errores masivos en bosque/clase 6, pero la clase 19 merece revisión.

## Decisión

- **H6 piloto:** consistente en 53/60 (88.3 %).
- **H6 completo:** pendiente de revisar los 240 puntos restantes.
- No se cambia el modelo ni se agrega ninguna variable.
