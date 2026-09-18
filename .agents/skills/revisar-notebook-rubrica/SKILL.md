---
name: revisar-notebook-rubrica
description: Audita el notebook ejecutado del Microproyecto 2 contra la rúbrica oficial y produce una matriz de cumplimiento con evidencia y faltantes. Úsala antes de entregar o cuando se solicite una revisión; no modifica archivos salvo autorización posterior.
---

# Revisar notebook contra la rúbrica

Realiza una auditoría de solo lectura antes de proponer cambios.

## Entradas

- `references/Microproyecto2.pdf`.
- Notebook objetivo, preferiblemente ejecutado.
- `CHECKLIST.md` como control secundario, nunca como sustituto de la rúbrica.

## Revisión

1. Extrae cada criterio, porcentaje y requisito explícito del enunciado.
2. Localiza evidencia concreta en celdas Markdown, código y salidas.
3. Clasifica cada requisito como `Cumple`, `Parcial` o `No cumple`.
4. Señala problemas de fuga de datos, reproducibilidad, métricas, justificación y celdas sin ejecutar.
5. Verifica específicamente los cinco tópicos interpretados y las cuatro predicciones sobre test.
6. Entrega primero una tabla con criterio, peso, estado, evidencia y corrección necesaria.
7. Prioriza los faltantes por impacto estimado en la nota.

No modifiques el notebook durante la auditoría. Solicita autorización antes de implementar correcciones.

