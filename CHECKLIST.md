# Checklist de la rúbrica

Actualizar la columna Estado únicamente con evidencia visible en el notebook ejecutado.

| Criterio | Peso | Evidencia esperada | Estado |
|---|---:|---|---|
| Preparación y reducción de dimensionalidad | 30% | Auditoría, limpieza justificada, TF-IDF, división sin fuga y SVD | Pendiente |
| Pipeline de preparación | 15% | Pipeline reproducible con transformaciones ajustadas dentro de CV | Pendiente |
| Clasificación, búsqueda y evaluación | 30% | Algoritmo justificado, hiperparámetros, CV y métricas apropiadas | Pendiente |
| Desempeño en textos no vistos | 10% | Tabla con al menos cuatro textos de test, real y predicho | Pendiente |
| LSA e interpretación | 15% | 10-20 componentes, palabras principales y cinco tópicos interpretados | Pendiente |
| Streamlit opcional | +15 | Entrada libre, mismo pipeline y predicción funcional | No iniciado |

## Controles metodológicos

- [ ] Se verificaron nulos y duplicados.
- [ ] Se describió la distribución de ODS y la ausencia del ODS 17.
- [ ] Se analizaron longitudes de los textos.
- [ ] El split se realizó antes de ajustar TF-IDF/SVD.
- [ ] El test no se utilizó para selección de modelo.
- [ ] La búsqueda usa validación cruzada estratificada.
- [ ] Macro-F1 es la métrica principal y está justificada.
- [ ] Se incluyeron matriz de confusión y reporte por clase.
- [ ] Cada visualización importante tiene interpretación.
- [ ] El notebook se ejecuta de principio a fin con kernel reiniciado.
- [ ] El `.html` conserva todas las salidas.
- [ ] Existe una conclusión consolidada.

