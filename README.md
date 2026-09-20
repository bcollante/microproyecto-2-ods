# Microproyecto 2 — Clasificación de textos según los ODS

Solución de NLP + machine learning que clasifica textos en español según los
Objetivos de Desarrollo Sostenible (ODS) de la ONU, desarrollada para el curso
*Machine Learning No Supervisado* de la Maestría en Inteligencia Artificial.

Método: limpieza → bolsa de palabras con TF-IDF → análisis semántico latente
(`TruncatedSVD`) → clasificador lineal con búsqueda de hiperparámetros, todo
dentro de un `Pipeline` de scikit-learn. Incluye una app Streamlit que reutiliza
el mismo pipeline.

## Entregables (solo se adjuntan dos archivos)

- `entregables/microproyecto2_ods.ipynb` — notebook ejecutado de punta a punta,
  **autocontenido** (todas las funciones están en sus celdas; solo necesita el
  Excel del dataset junto a él o en `data/`).
- `entregables/streamlit_app.py` — app Streamlit **autocontenida**; necesita el
  `pipeline_ods.joblib` que genera el notebook (lo busca junto a ella o en
  `models/`, o se sube desde la barra lateral).
- `entregables/microproyecto2_ods.html` — exportación con todas las salidas
  (por si se pide).

## Reproducir

```powershell
.\setup.ps1                                   # crea .venv, instala requirements.txt y registra el kernel
.\.venv\Scripts\python.exe scripts\run_notebook.py   # ejecuta el notebook y copia notebook, HTML y app a entregables/ (≈ 10 min)
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py   # app (usa models/pipeline_ods.joblib)
```

## Estructura

- `data/Datos_textosODS.xlsx`: dataset del proyecto (9 656 textos, ODS 1–16).
- `notebooks/`: notebook de trabajo (fuente).
- `app/streamlit_app.py`: aplicación Streamlit (bono), autocontenida.
- `models/`: pipeline serializado (no se versiona; lo genera el notebook).
- `outputs/`: figuras y tablas generadas por el notebook (no se versionan).
- `entregables/`: versiones finales `.ipynb`, `.html` y `streamlit_app.py`.
- `scripts/run_notebook.py`: ejecuta el notebook con kernel limpio (evitando la suspensión del equipo) y copia `.ipynb`, `.html` y la app a `entregables/`.
- `docs/bitacora_desarrollo.md`: qué faltaba, qué se hizo y por qué, etapa por etapa.
- `CHECKLIST.md`: matriz de cumplimiento de la rúbrica con evidencia.
- `references/`: enunciado (`Microproyecto2.pdf`) y tutoriales del curso.
- `AGENTS.md` / `CLAUDE.md`: reglas de trabajo para agentes de código.
