---
name: microproyecto-mlns
description: Desarrolla por etapas el Microproyecto 2 de clasificación de textos ODS en Jupyter, alineando preparación, LSA, pipeline, clasificación y evaluación con su rúbrica. Úsala al implementar o revisar una etapa de este repositorio; no la uses para proyectos ajenos.
---

# Microproyecto MLNS

Lee primero `AGENTS.md`, `CHECKLIST.md` y la sección aplicable de `references/Microproyecto2.pdf`.

## Procedimiento

1. Identifica la etapa solicitada y el criterio exacto de la rúbrica que cubre.
2. Inspecciona el notebook y funciones relacionadas antes de editar.
3. Propón un cambio pequeño con una evidencia observable.
4. Implementa sin adelantar etapas no solicitadas.
5. Ejecuta o valida el cambio y comunica el resultado real.
6. Actualiza `CHECKLIST.md` solo cuando exista evidencia visible y reproducible.

## Decisiones esenciales

- Conserva un test aislado y evita fuga de información.
- Ajusta TF-IDF, SVD y clasificador dentro de pipelines durante validación.
- Usa macro-F1 como selección principal y acompáñala con métricas complementarias.
- Separa el LSA interpretativo de 10-20 componentes del SVD que se ajuste para clasificación cuando sea metodológicamente conveniente.
- Interpreta resultados en Markdown; no dejes tablas o gráficos sin explicación.
- Expresa el alcance como ODS 1-16 mientras el dataset no contenga ODS 17.
- No uses el tutorial de Isolation Forest para resolver esta clasificación.

## Criterio de finalización

Una etapa está completa cuando el código es reproducible, la salida es visible, la decisión está justificada y la evidencia puede localizarse en el notebook.

