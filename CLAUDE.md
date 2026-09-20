# CLAUDE.md — Microproyecto 2: clasificación de textos ODS

Contexto para trabajar en este repo con Claude Code. Microproyecto académico
(Maestría en IA, curso *Machine Learning No Supervisado*). El enunciado y la
rúbrica oficial están en `Microproyecto2.pdf` (copia en
`references/Microproyecto2.pdf`). Las reglas metodológicas detalladas están en
`AGENTS.md`; este archivo las resume y añade lo operativo.

## Objetivo

Clasificar automáticamente textos en español según los Objetivos de Desarrollo
Sostenible (ODS) de la ONU con: bolsa de palabras + TF-IDF, análisis semántico
latente (LSA vía `TruncatedSVD`) y un clasificador supervisado con búsqueda de
hiperparámetros; evaluación sobre test no visto; app Streamlit opcional (bono).

## Estructura real del repo

```
data/Datos_textosODS.xlsx        dataset entregado (NO mover ni renombrar)
notebooks/microproyecto2_ods.ipynb   notebook de trabajo (fuente). AUTOCONTENIDO: todas las
                                     funciones auxiliares viven en sus celdas (no importa src/)
app/streamlit_app.py   app Streamlit (bono). AUTOCONTENIDA: define clean_text y ODS_NAMES,
                       busca pipeline_ods.joblib junto a ella / models/ / ../models/ o permite subirlo
entregables/           microproyecto2_ods.{ipynb,html} + streamlit_app.py (lo que se entrega)
models/pipeline_ods.joblib   pipeline final serializado (gitignored; lo genera el notebook)
outputs/figures, outputs/tables   salidas del notebook (gitignored)
docs/bitacora_desarrollo.md  bitácora: qué faltaba, qué se hizo y por qué, por etapa
scripts/run_notebook.py  ejecuta el notebook (sin dejar dormir el equipo) y exporta entregables
CHECKLIST.md           matriz de cumplimiento de la rúbrica
references/            enunciado + tutoriales del curso
```

## Dataset (verificado)

- 9 656 filas, columnas `textos` (str) y `ODS` (int). Sin nulos ni duplicados.
- **Solo aparecen las etiquetas 1–16**; ODS 17 no existe en los datos (sí en
  `ODS_NAMES` para que la UI sea correcta). Nunca afirmar que el modelo cubre
  el ODS 17.
- Desbalance moderado: 312 (ODS 12) a 1 080 (ODS 16) textos por clase → métrica
  principal **F1 macro** y `class_weight="balanced"`.
- Textos de ~110 palabras en promedio (traducidos con DeepL, algunos
  aumentados con ChatGPT).

## Las 4 actividades del enunciado y dónde viven

| Actividad | Sección del notebook | Funciones (definidas en esa sección) |
|---|---|---|
| 1. BOW + TF-IDF en un pipeline | 6 | `clean_text`, `get_spanish_stopwords`, `build_tfidf_vectorizer` |
| 2. LSA con TruncatedSVD (10–20 comp.), ≥5 tópicos interpretados | 7 | `fit_lsa`, `top_terms_per_component`, `component_scores_by_label` |
| 3. Clasificador con reducción de dimensionalidad + búsqueda de hiperparámetros, justificado | 8–9 | `build_classification_pipeline`, `candidate_search_spaces`, `run_grid_search` |
| 4. Evaluación en test no visto, ≥4 textos mostrados | 10–11 | `compute_metrics`, `per_class_report`, `plot_confusion_matrix`, `sample_predictions_table` |

## Rúbrica

| Criterio | % |
|---|---|
| Preparación de datos + reducción de dimensionalidad, justificada | 30 |
| Construcción del pipeline de preparación | 15 |
| Clasificador con búsqueda de hiperparámetros, métricas y algoritmo justificados | 30 |
| Evidencia sobre ≥4 textos de test no usados en entrenamiento | 10 |
| LSA + interpretación cualitativa de ≥5 tópicos vs. ODS | 15 |
| **Bono:** app Streamlit funcional que reutiliza el mismo pipeline | +15 |

## Reglas de trabajo

- `train_test_split(test_size=0.2, stratify=y, random_state=42)` en la sección 5
  del notebook. El test es sagrado: nunca se usa para ajustar TF-IDF/SVD, elegir
  modelo ni hiperparámetros. Se evalúa **una sola vez** al final.
- TF-IDF, SVD y clasificador siempre dentro de un `Pipeline`, reajustado en
  cada fold de la validación cruzada (`StratifiedKFold(5, shuffle=True)`).
- El LSA interpretativo usa 15 componentes (rango 10–20 exigido). El SVD del
  clasificador puede usar más componentes (se busca en {100, 200, 300}) — son
  dos usos distintos de la misma técnica y el notebook lo explica.
- Cada decisión (limpieza, parámetros TF-IDF, algoritmo, métrica) va
  justificada en una celda markdown junto al código — es criterio de rúbrica.
- **Solo se pueden entregar dos archivos: el notebook y `streamlit_app.py`.** Por
  eso ambos son autocontenidos y NO existe paquete `src/`. `clean_text` y
  `ODS_NAMES` están duplicados a propósito en notebook y app: si se cambia uno
  hay que cambiar el otro (la app registra `clean_text` en `__main__` y en un
  shim `src.preprocessing` para poder des-serializar el pipeline).
- El notebook localiza el dataset en `data/`, `../data/` o junto a él
  (`Datos_textosODS.xlsx` / `textos_ods.xlsx`) y escribe `outputs/` y
  `models/` en la raíz detectada.
- Tras editar el notebook, reejecutarlo completo con kernel limpio y regenerar
  el HTML (ver abajo). No entregar sin outputs visibles.
- Registrar cada cambio relevante en `docs/bitacora_desarrollo.md` y reflejar
  evidencia en `CHECKLIST.md`.
- No usar Isolation Forest (el tutorial en `references/` es de otro tema).
- Si el equipo se suspende durante `nbconvert --execute`, la celda del grid
  agota el timeout: usar `scripts/run_notebook.py`, que lo impide.

## Entorno y comandos

Python 3.14, venv en `.venv/` (creado con `py -3.14 -m venv .venv`;
`setup.ps1` hace lo mismo y registra el kernel). Todas las dependencias de
`requirements.txt` instalaron con wheels para 3.14 en Windows.

```powershell
# ejecutar el notebook completo con kernel limpio y exportar entregables (≈ 10–15 min)
# (impide que Windows suspenda el equipo durante el grid search)
.\.venv\Scripts\python.exe scripts\run_notebook.py

# equivalente manual
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=3600 notebooks\microproyecto2_ods.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to html notebooks\microproyecto2_ods.ipynb --output-dir entregables
Copy-Item notebooks\microproyecto2_ods.ipynb entregables\

# app Streamlit (requiere pipeline_ods.joblib generado por el notebook; lo busca en models/)
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

`scripts/run_notebook.py` ejecuta el notebook, exporta el HTML y copia
notebook + app a `entregables/`. La app funciona suelta en cualquier carpeta
siempre que `pipeline_ods.joblib` (lo genera el notebook, ~65 MB) esté junto a
ella o se cargue desde la barra lateral.

## Resultados de referencia

Ver `docs/bitacora_desarrollo.md` § 4 y la sección 10 del notebook ejecutado.
Son números reales de la última ejecución, no metas.
