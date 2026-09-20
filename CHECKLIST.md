# Checklist de la rúbrica

Actualizar la columna Estado únicamente con evidencia visible en el notebook ejecutado
(`entregables/microproyecto2_ods.ipynb` / `.html`). Última actualización: 2026-09-18.

| Criterio | Peso | Evidencia esperada | Estado | Evidencia en el notebook |
|---|---:|---|---|---|
| Preparación y reducción de dimensionalidad | 30% | Auditoría, limpieza justificada, TF-IDF, división sin fuga y SVD | Cumple | §3 auditoría (nulos, duplicados, etiquetas 1–16); §4 EDA con interpretación; §5 split estratificado antes de ajustar nada; §6 tablas de justificación de limpieza y de cada parámetro TF-IDF; §7–8 SVD justificado (LSA interpretativo vs. SVD del clasificador) |
| Pipeline de preparación | 15% | Pipeline reproducible con transformaciones ajustadas dentro de CV | Cumple | §6 `build_tfidf_vectorizer` (limpieza `clean_text` como `preprocessor`) + §8 `build_classification_pipeline` (`tfidf -> svd -> norm -> clf`), ambas definidas en el notebook; §8 muestra el `Pipeline`; §9 lo reajusta en cada fold de `GridSearchCV` |
| Clasificación, búsqueda y evaluación | 30% | Algoritmo justificado, hiperparámetros, CV y métricas apropiadas | Cumple | §8 justificación de SVD, `Normalizer`, modelos lineales, `class_weight` y métricas; §9 `GridSearchCV` 5-fold estratificado sobre LogReg y LinearSVC (90 ajustes), tabla + gráfico + interpretación; §10 métricas globales, reporte por clase, matriz de confusión, pares confundidos, interpretación |
| Desempeño en textos no vistos | 10% | Tabla con al menos cuatro textos de test, real y predicho | Cumple | §11: 8 textos aleatorios de test + 5 errores informativos, con fragmento, ODS real, ODS predicho, acierto y top-3 por margen; §13: 4 textos nuevos escritos a mano |
| LSA e interpretación | 15% | 10-20 componentes, palabras principales y cinco tópicos interpretados | Cumple | §7: comparación k = 10/15/20, LSA de 15 comp., tabla y gráfico de 12 términos por componente, heatmap componente × ODS, tabla de interpretación de 12 componentes (≥ 8 asociadas a un ODS concreto) |
| Streamlit opcional | +15 | Entrada libre, mismo pipeline y predicción funcional | Cumple | `streamlit_app.py` (autocontenida): texto libre, carga `pipeline_ods.joblib` (mismo `Pipeline` serializado en §13), predice ODS + top-3 + gráfico; verificado con `python -m streamlit run app/streamlit_app.py` |

## Controles metodológicos

- [x] Se verificaron nulos y duplicados (§3: 0 nulos, 0 duplicados).
- [x] Se describió la distribución de ODS y la ausencia del ODS 17 (§3, §4).
- [x] Se analizaron longitudes de los textos (§4: histograma y boxplot por ODS).
- [x] El split se realizó antes de ajustar TF-IDF/SVD (§5 precede a §6–§9).
- [x] El test no se utilizó para selección de modelo (solo aparece en §10, §11 y la verificación de §13).
- [x] La búsqueda usa validación cruzada estratificada (`StratifiedKFold(5, shuffle=True, random_state=42)`).
- [x] Macro-F1 es la métrica principal y está justificada (§4 y §8).
- [x] Se incluyeron matriz de confusión y reporte por clase (§10).
- [x] Cada visualización importante tiene interpretación (celda markdown tras cada figura/tabla).
- [x] El notebook se ejecuta de principio a fin con kernel reiniciado (`scripts/run_notebook.py` → `nbconvert --execute`).
- [x] El `.html` conserva todas las salidas (`entregables/microproyecto2_ods.html`).
- [x] Existe una conclusión consolidada (§12: resultados, limitaciones, mejoras).
