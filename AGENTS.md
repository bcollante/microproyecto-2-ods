# Instrucciones del proyecto

## Objetivo

Desarrollar una solución reproducible para clasificar textos en español según los Objetivos de Desarrollo Sostenible presentes en el conjunto de datos del Microproyecto 2.

## Fuentes de verdad

1. `references/Microproyecto2.pdf`: enunciado y rúbrica oficial.
2. `data/Datos_textosODS.xlsx`: conjunto de datos entregado.
3. Los tutoriales dentro de `references/`: material de apoyo del curso.
4. Este archivo: reglas operativas del repositorio.

Si existe una contradicción, prevalecen el enunciado y la rúbrica.

## Alcance actual

- Preparar la estructura y desarrollar el proyecto por etapas verificables.
- El dataset contiene etiquetas ODS 1 a 16. No afirmar que el modelo cubre el ODS 17 mientras no existan ejemplos de esa clase.
- La aplicación Streamlit es opcional y se aborda únicamente después de completar y validar el notebook obligatorio.

## Entregables obligatorios

- `entregables/microproyecto2_ods.ipynb`, completamente ejecutado.
- `entregables/microproyecto2_ods.html`, exportado con todas las salidas visibles.
- Preparación de textos con bolsa de palabras y ponderación TF-IDF.
- Pipeline de preparación y clasificación.
- Modelo LSA sobre TF-IDF con interpretación cualitativa de al menos cinco tópicos.
- Reducción de dimensionalidad dentro del flujo de clasificación.
- Búsqueda de hiperparámetros y validación con métricas justificadas.
- Evaluación final sobre test aislado.
- Evidencia de al menos cuatro predicciones sobre textos no utilizados en entrenamiento.

## Reglas metodológicas obligatorias

- Separar entrenamiento y test antes de ajustar TF-IDF, SVD o cualquier transformación aprendida.
- Utilizar división estratificada y `random_state=42`, salvo justificación explícita.
- Ajustar transformaciones y modelo dentro de un `Pipeline` durante validación cruzada.
- No utilizar el conjunto test para seleccionar transformaciones, modelos ni hiperparámetros.
- Usar macro-F1 como métrica principal por el desbalance entre clases.
- Reportar también accuracy, precision macro, recall macro, F1 weighted, reporte por clase y matriz de confusión.
- Mantener un LSA exploratorio de 10 a 20 componentes para tópicos. La clasificación puede evaluar otro número de componentes SVD.
- Explicar que los componentes LSA no tienen por qué corresponder uno a uno con los ODS.
- Revisar nulos, duplicados exactos, distribución de clases y longitudes antes de modelar.
- No inventar métricas, resultados, interpretaciones ni ejecuciones.

## Estilo del notebook

- Alternar celdas Markdown y código para que cada decisión tenga justificación.
- Después de cada tabla o gráfico relevante, escribir una interpretación breve y específica.
- Usar nombres claros y consistentes; preferir funciones reutilizables a bloques repetidos.
- Incluir títulos, ejes, leyendas y unidades cuando correspondan.
- Mantener visibles las salidas finales de todas las celdas requeridas.
- Terminar con una sola conclusión consolidada: resultados, limitaciones y mejoras.

## Flujo de trabajo para Codex

1. Antes de editar, leer la sección correspondiente del enunciado y revisar el estado actual del notebook.
2. Explicar brevemente qué se modificará y qué criterio de la rúbrica cubre.
3. Realizar cambios pequeños y verificables.
4. Ejecutar o validar únicamente la etapa modificada.
5. No avanzar a la etapa siguiente si la actual presenta errores.
6. Al terminar cada etapa, actualizar `CHECKLIST.md` con evidencia verificable.
7. No exportar entregables finales hasta que todas las celdas se ejecuten desde un kernel reiniciado.

## Restricciones de edición

- Preservar resultados válidos y cambios del estudiante.
- No sustituir decisiones metodológicas sin explicar el motivo.
- No agregar algoritmos que no contribuyan a la rúbrica.
- No usar Isolation Forest para la clasificación ODS; el tutorial adjunto pertenece a otro tema del curso.
- No desarrollar todo el proyecto en una sola intervención sin puntos de revisión.

