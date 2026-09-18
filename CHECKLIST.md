# Checklist de la rúbrica

Actualizar la columna Estado únicamente con evidencia visible en el notebook ejecutado.

## Avance por fases

- [x] Fase 1: contexto, problema, objetivo y alcance ODS 1-16 documentados en la sección 1 de `notebooks/microproyecto2_ods.ipynb`.
- [x] Fase 2: carga y auditoría del conjunto de datos en la sección 3 de `notebooks/microproyecto2_ods.ipynb` (salida de la celda y lectura de resultados).
- [x] Fase 3: análisis exploratorio en la sección 4 (conteos, longitudes y ejemplos, con interpretación).
- [x] Fase 4: división estratificada en la sección 5 (7.724 train / 1.932 test; índices disjuntos).
- [x] Fase 5: TF-IDF inicial en la sección 6 (pipeline ajustado solo sobre train; matriz de 7.724 x 20.000).
- [x] Fase 6: LSA exploratorio de 12 componentes e interpretación de al menos cinco en la sección 7.
- [x] Fase 7: pipeline TF-IDF + SVD + normalización + regresión logística y búsqueda estratificada en las secciones 8 y 9 (macro-F1 CV 0,822).
- [x] Fase 8: evaluación única del modelo elegido sobre 1.932 textos de test en la sección 10 (macro-F1 0,827 y matriz de confusión).
- [x] Fase 9: cuatro predicciones de test con ODS real, predicho e interpretación en la sección 11.
- [x] Conclusión consolidada en la sección 12: resultados, limitaciones y mejoras basadas en la evidencia ejecutada.

| Criterio | Peso | Evidencia esperada | Estado |
|---|---:|---|---|
| Preparación y reducción de dimensionalidad | 30% | Auditoría, limpieza justificada, TF-IDF, división sin fuga y SVD | Evidencia en secciones 3-8 |
| Pipeline de preparación | 15% | Pipeline reproducible con transformaciones ajustadas dentro de CV | Evidencia en secciones 8-9 |
| Clasificación, búsqueda y evaluación | 30% | Algoritmo justificado, hiperparámetros, CV y métricas apropiadas | Evidencia en secciones 8-10 |
| Desempeño en textos no vistos | 10% | Tabla con al menos cuatro textos de test, real y predicho | Evidencia en sección 11 |
| LSA e interpretación | 15% | 10-20 componentes, palabras principales y cinco tópicos interpretados | Evidencia en sección 7 |
| Streamlit opcional | +15 | Entrada libre, mismo pipeline y predicción funcional | No iniciado |

## Controles metodológicos

- [x] Se verificaron nulos y duplicados exactos: ambos 0 en la sección 3 del notebook.
- [x] Se describió la distribución de ODS y la ausencia del ODS 17 en la sección 3 del notebook.
- [x] Se analizaron longitudes de los textos en la sección 4.
- [x] El split se realizó antes de ajustar TF-IDF y SVD.
- [x] El test no se utilizó para selección de modelo; se evaluó tras elegir el candidato por CV.
- [x] La búsqueda usa validación cruzada estratificada de tres pliegues.
- [x] Macro-F1 es la métrica principal, justificada por el desbalance y usada en CV.
- [x] Se incluyeron matriz de confusión y reporte por clase en la sección 10.
- [x] Las tablas y la matriz de confusión importantes tienen interpretación.
- [ ] El notebook se ejecuta de principio a fin con kernel reiniciado.
- [ ] El `.html` conserva todas las salidas.
- [x] Existe una conclusión consolidada al final del notebook, en la sección 12.
