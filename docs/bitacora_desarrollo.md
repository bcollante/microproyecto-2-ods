# Bitácora de desarrollo — Microproyecto 2 (clasificación de textos ODS)

Documento de trabajo que registra **qué faltaba** en el repo al iniciar, **qué se
hizo** en cada etapa y **por qué**, con referencia al criterio de la rúbrica de
`Microproyecto2.pdf` que cubre cada cambio. Se actualiza etapa por etapa.

Fecha de inicio: 2026-09-18.

---

## 0. Diagnóstico inicial (estado del repo antes de intervenir)

| Componente | Estado encontrado | Impacto en la rúbrica |
|---|---|---|
| `notebooks/microproyecto2_ods.ipynb` | Esqueleto de 24 celdas; **todas** las celdas de código eran comentarios `# Pendiente: …`, ninguna ejecutada. | Ningún criterio cubierto (0 %). |
| `src/preprocessing.py`, `src/topics.py`, `src/modeling.py`, `src/evaluation.py` | Solo docstrings de una línea, sin funciones. | Sin pipeline reutilizable (15 %). |
| `app/streamlit_app.py` | Docstring vacío. | Bono Streamlit no iniciado. |
| `entregables/` | Solo `.gitkeep`; sin `.ipynb` ni `.html`. | Entregable inexistente. |
| `models/`, `outputs/figures`, `outputs/tables` | Solo `.gitkeep`. | — |
| `CHECKLIST.md` | Todos los criterios en `Pendiente` / `No iniciado`. | — |
| `CLAUDE.md` | Sin trackear en git y **desactualizado**: refería rutas inexistentes (`data/raw/textos_ods.xlsx`, `src/ods_catalog.py`, `notebooks/01_clasificacion_ods.ipynb`, `app/app.py`) y resultados de referencia de otra versión del proyecto. | Confundía a cualquier agente/persona que trabajara el repo. |
| `.venv/` | No existía dentro del repo (`setup.ps1` lo asume). El venv del directorio padre no tenía `scikit-learn`/`pandas`. | El notebook no podía ejecutarse. |
| `data/Datos_textosODS.xlsx` | Presente y correcto: 9 656 filas, columnas `textos` y `ODS`, sin nulos ni duplicados, etiquetas 1–16 (ODS 17 ausente). | — |
| `references/` | Enunciado y tutoriales del curso presentes. | — |

**Conclusión del diagnóstico:** el repo tenía una buena *estructura* (carpetas,
reglas en `AGENTS.md`, checklist, skills) pero **cero implementación**. Todo el
contenido evaluable había que construirlo.

Datos verificados en la auditoría inicial (sirven de base a las decisiones):

- Longitud de textos: media ≈ 709 caracteres (≈ 111 palabras), mín. 143, máx. 1 977.
- Distribución por clase: de 312 (ODS 12) a 1 080 (ODS 16) textos → ratio 3.5:1
  entre la clase más grande y la más pequeña → **desbalance moderado**.

---

## 1. Entorno reproducible

**Qué faltaba:** no había venv del proyecto y por tanto no se podía ejecutar nada.

**Qué se hizo:**

- Se creó `.venv/` con Python 3.14 (`py -3.14 -m venv .venv`) e instaló
  `requirements.txt` sin cambios (scikit-learn 1.9.1, pandas 2.3.3, nltk 3.10.3,
  streamlit 1.64.0). Todos los wheels disponibles para 3.14 en Windows.
- Se verificó que el corpus de stopwords en español de NLTK está disponible;
  `src/preprocessing.get_spanish_stopwords` lo descarga automáticamente si falta.

---

## 2. Módulos `src/` (pipeline reutilizable) — rúbrica: *pipeline de preparación* (15 %)

**Qué faltaba:** los cuatro módulos estaban vacíos. Sin ellos, notebook y app
tendrían que duplicar lógica (riesgo de que la app no use "el mismo pipeline",
condición explícita del bono).

**Qué se implementó y por qué:**

### `src/preprocessing.py`
- `ODS_NAMES` (1–17) y `ods_label()`: catálogo único de nombres, usado en
  tablas, gráficos y en la app.
- `load_dataset()`: carga y valida columnas.
- `clean_text()`: minúsculas, quita URLs, normaliza Unicode NFC, elimina
  dígitos/puntuación **conservando tildes y ñ**, colapsa espacios.
  *Decisión:* no se aplica stemming/lematización para mantener legibles los
  términos de los tópicos LSA y los coeficientes del clasificador; el corpus es
  formal y traducido de forma homogénea, así que la ganancia de un stemmer sería
  marginal frente a la pérdida de interpretabilidad.
- `get_spanish_stopwords()`: NLTK + lista corta de términos de dominio muy
  frecuentes y no discriminantes ("puede", "además", "ejemplo", …).
- `build_tfidf_vectorizer()`: `preprocessor=clean_text` (la limpieza vive
  *dentro* del pipeline), unigramas + bigramas, `min_df=3`, `max_df=0.9`,
  `sublinear_tf=True`, `max_features=30 000`, tokens de ≥ 3 letras.
  Justificación de cada parámetro en el docstring y en el notebook (sección 6).

### `src/topics.py`
- `fit_lsa()`, `explained_variance_table()`, `top_terms_per_component()`
  (ordena por |peso| y conserva el signo, porque una componente LSA contrasta
  dos polos), `topics_summary()`, `component_scores_by_label()` y
  `dominant_ods_per_component()` — estas dos últimas permiten **contrastar la
  interpretación cualitativa con la etiqueta real** (puntuación media de cada
  componente por ODS).

### `src/modeling.py`
- `build_classification_pipeline()`: `tfidf → svd → norm(L2) → clf`.
- `candidate_search_spaces()`: rejillas para `LogisticRegression` y `LinearSVC`
  (ambos `class_weight="balanced"`), variando `svd__n_components ∈ {100, 200, 300}`
  y `C`.
- `run_grid_search()`: `GridSearchCV` con `StratifiedKFold(5, shuffle=True)`,
  `refit="f1_macro"`, también registra accuracy.
- `cv_results_table()`: tabla legible ordenada por F1 macro.

### `src/evaluation.py`
- `compute_metrics()` (accuracy, precision/recall/F1 macro, F1 weighted),
  `per_class_report()`, `plot_confusion_matrix()` (normalizada por fila),
  `most_confused_pairs()`, `sample_predictions_table()` (evidencia sobre test
  con top-3 probabilidades).

---

## 3. Exploración previa (resultados reales que guían el notebook)

Ejecutada con scripts temporales sobre el split de entrenamiento (80 %,
estratificado, `random_state=42`); el test **no se tocó**.

### LSA (15 componentes sobre TF-IDF de entrenamiento, 7 724 × 26 448)

Varianza explicada acumulada por 15 componentes ≈ 3.1 % — valor bajo pero
normal en matrices TF-IDF dispersas de vocabulario amplio; lo relevante es la
interpretabilidad de los ejes, no la varianza.

Componentes claramente asociadas a un ODS (términos con mayor peso positivo →
ODS con mayor puntuación media):

| Comp. | Términos principales | ODS dominante (media) |
|---|---|---|
| C2 | derechos, derechos humanos, internacional, artículo, ley, justicia | ODS 16 (0.129) |
| C4 | estudiantes, educación, escuelas, aprendizaje, docentes, primaria | ODS 4 (0.093) |
| C5 | mujeres, género, hombres, igualdad, igualdad género, participación | ODS 5 (0.069) |
| C6 | salud, atención, servicios, mental, salud mental, atención primaria, pacientes | ODS 3 (0.073) |
| C7 | agua, aguas subterráneas, aguas residuales, hídricos, agua potable | ODS 6 (0.112) |
| C8 | energía, electricidad, renovables, energías renovables (polo −: cambio climático) | ODS 7 (0.063) |
| C9 | países ocde, cambio climático, promedio (polo −: pobreza, hogares) | ODS 13 (0.062) |
| C10 (polo −) | empleo, laboral, trabajadores, mercado laboral | ODS 1 / ODS 8 |
| C11 | transporte, ciudades, rurales, zonas, público | ODS 11 (0.042), ODS 9 |
| C13 | producción, alimentos, alimentaria, precios, pesca, seguridad alimentaria | ODS 2 (0.067), ODS 14, ODS 12 |

C1 es el eje "genérico" (países, desarrollo, políticas) típico de la primera
componente de LSA. Se dispone de más de 5 tópicos interpretables → cumple el
15 % de la rúbrica.

### Búsqueda de hiperparámetros (CV 5-fold estratificada, F1 macro, solo train)

| Configuración | F1 macro CV | Nota |
|---|---|---|
| TF-IDF → LogReg **sin SVD** (referencia) | 0.870 ± 0.007 | cota superior |
| TF-IDF → SVD(15) → LogReg (LSA interpretativo) | 0.715 ± 0.008 | 15 comp. no bastan para 16 clases |
| LogReg, mejor: C=1, 300 comp. | 0.844 ± 0.004 | fit ≈ 165 s |
| **LinearSVC, mejor: C=0.3, 300 comp.** | **0.849 ± 0.006** | fit ≈ 77 s → **seleccionado** |

Tendencias: más componentes ⇒ mejor (100→200→300 monótono en ambos); gana el
`C` más pequeño (mayor regularización). LinearSVC supera a LogReg en todas las
configuraciones equivalentes y entrena 2× más rápido.

---

## 4. Notebook `notebooks/microproyecto2_ods.ipynb` (todos los criterios)

**Qué faltaba:** las 13 secciones eran encabezados con "Pendiente".

**Qué se hizo:** se regeneró el notebook completo (50 celdas, markdown + código
alternados) manteniendo la numeración de secciones del esqueleto original:

| Sección | Contenido | Criterio de rúbrica |
|---|---|---|
| 1 | Contexto, problema de ML, alcance real (ODS 1–16), objetivos | — |
| 2 | Imports desde `src/`, `RANDOM_STATE=42` | reproducibilidad |
| 3 | Carga y auditoría: shape, tipos, nulos, duplicados, etiquetas ausentes | Preparación (30 %) |
| 4 | EDA: distribución por clase (gráfico + tabla), longitud (hist + boxplot), ejemplos; cada gráfico con interpretación | Preparación (30 %) |
| 5 | `train_test_split(0.2, stratify, 42)` + tabla de proporciones train/test | sin fuga |
| 6 | Tabla de decisiones de limpieza y de parámetros TF-IDF con justificación; demo de `clean_text`; matriz TF-IDF de train; términos top | Preparación + Pipeline (30 % + 15 %) |
| 7 | LSA: comparación k=10/15/20, varianza explicada, tabla y gráfico de términos por componente, heatmap componente×ODS, **tabla de interpretación de 12 componentes** con evidencia cuantitativa | LSA (15 %) |
| 8 | Arquitectura del pipeline; justificación de SVD como reducción, del `Normalizer`, de los algoritmos lineales, de `class_weight` y de las métricas; dos referencias (SVD-15 y sin SVD) | Clasificación (30 %) |
| 9 | `GridSearchCV` para LogReg y LinearSVC (90 ajustes), tabla y gráfico F1 vs componentes, selección automática del mejor e interpretación | Clasificación (30 %) |
| 10 | Evaluación única en test: métricas globales, reporte por clase (tabla + gráfico), matriz de confusión normalizada, pares más confundidos, interpretación | Clasificación (30 %) |
| 11 | 8 textos aleatorios de test + 5 errores "seguros", con fragmento, ODS real, predicho, acierto y top-3 por margen | Evidencia (10 %) |
| 12 | Conclusiones, limitaciones, mejoras | — |
| 13 | `joblib.dump` a `models/pipeline_ods.joblib`, verificación de recarga y predicción de 4 textos nuevos; instrucciones de la app | Bono |

Decisiones metodológicas clave (todas justificadas en el notebook):

- El LSA interpretativo (15 comp.) y el SVD del clasificador (300 comp.) son dos
  usos distintos de la misma técnica; el notebook lo explica y lo demuestra con
  la referencia SVD-15 (F1 0.715).
- Se eligió **LinearSVC** por F1 macro CV y velocidad; al no tener
  `predict_proba`, el ranking top-3 se hace con `decision_function` y se aclara
  que es un margen, no una probabilidad.
- Métrica principal F1 macro por desbalance; se reportan accuracy, precision y
  recall macro, F1 ponderado, reporte por clase y matriz de confusión.

---

## 5. Aplicación Streamlit (bono, +15)

**Qué faltaba:** `app/streamlit_app.py` solo tenía un docstring.

**Qué se implementó:**

- Carga `models/pipeline_ods.joblib` con `st.cache_resource`; si no existe,
  muestra un error claro indicando que hay que ejecutar el notebook.
- Área de texto libre + selector de ejemplos (educación, agua, energía, justicia)
  para probar rápidamente.
- Validación mínima (≥ 5 palabras tras `clean_text`).
- Predicción con el **mismo objeto `Pipeline`** entrenado en el notebook (la
  limpieza está dentro del `TfidfVectorizer` como `preprocessor`, por lo que no
  hay ningún paso manual que la app pueda hacer distinto).
- Muestra el ODS predicho con su nombre, top-3 con `st.metric`, gráfico de
  barras y tabla con todas las probabilidades (`predict_proba` si el
  clasificador lo soporta; si no, softmax de `decision_function` solo como
  ranking visual).
- Barra lateral con los pasos del pipeline, nº de componentes SVD, clasificador
  y las clases conocidas (explicita que ODS 17 no está).
- Expander que muestra el texto normalizado que realmente entra al TF-IDF.

Comando: `.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py`.

## 6. Documentación del repo

- `CLAUDE.md` reescrito con la **estructura real** (antes apuntaba a rutas
  inexistentes): mapa de archivos, dataset verificado, actividades ↔ secciones
  del notebook ↔ módulos `src/`, rúbrica, reglas y comandos exactos de
  ejecución/exportación.
- `README.md` reescrito con descripción del método, entregables, comandos de
  reproducción y estructura (antes empezaba sin título y sin instrucciones).
- `CHECKLIST.md` se actualiza al final con evidencia (sección/celda del notebook).

## 7. Ejecución del notebook — incidencias y solución

- **1.er intento** (`nbconvert --execute`, timeout 2 400 s por celda): la celda
  del grid search superó el timeout. Diagnóstico: un ajuste individual del
  pipeline más caro (SVD 300 comp. + LinearSVC) tarda ~12 s, así que los 90
  ajustes deberían tomar 10–15 min. Al medir tiempos individuales aparecieron
  ajustes de "3 horas": el **equipo entró en suspensión** durante la
  ejecución (el reloj de pared sigue corriendo y nbconvert agota el timeout).
- **Solución**: `scripts/run_notebook.py` ejecuta nbconvert llamando a
  `SetThreadExecutionState(ES_SYSTEM_REQUIRED)` para impedir que Windows
  suspenda el equipo mientras corre, y luego exporta el HTML a `entregables/`.
  Además, el grid usa `n_jobs=6` (con 12 workers había sobresuscripción de
  hilos BLAS y cada ajuste tardaba 5–10× más sin ganancia neta).
- Aviso de `loky resource_tracker` sobre carpetas temporales al cerrar el
  kernel: inofensivo, propio de joblib en Windows.

## 8. Resultados finales (ejecución completa, 2026-09-18, 536 s)

Ejecutado con `scripts/run_notebook.py`: 50 celdas, 23 de código, **0 errores**.

| Métrica (test, 1 932 textos nunca vistos) | Valor |
|---|---|
| Accuracy | 0.885 |
| Precision macro | 0.853 |
| Recall macro | 0.855 |
| **F1 macro** | **0.854** |
| F1 ponderado | 0.885 |

- CV (train): F1 macro 0.849 ± 0.006, accuracy 0.874 → sin sobreajuste al
  procedimiento de selección.
- Modelo final: `LinearSVC(C=0.3, class_weight="balanced")` sobre 300
  componentes SVD (grid de 90 ajustes; LogReg mejor = 0.844).
- Mejores clases (F1 > 0.9): ODS 16, 4, 5, 3, 6, 7, 14. Peores: ODS 8 (0.59),
  10 (0.68), 9 (0.74) — bloque temático trabajo/desigualdad/pobreza.
- Evidencia: 8/8 aciertos en la muestra aleatoria de test; en 3 de los 5
  errores "seguros" el ODS real es la 2.ª opción del ranking.
- Tras la ejecución se ajustaron **solo celdas markdown** de las secciones
  10–12 para que la interpretación cite las cifras reales (clases difíciles,
  pares confundidos, efecto del desbalance); no requiere reejecutar.

**Entregables generados**: `entregables/microproyecto2_ods.ipynb` (1.07 MB) y
`entregables/microproyecto2_ods.html` (1.39 MB), con todas las salidas.
`models/pipeline_ods.joblib` (64.8 MB, gitignored).

**App Streamlit** verificada en navegador con el modelo real: carga el
pipeline (SVD 300, LinearSVC, clases 1–16), clasifica texto libre (ejemplo de
brecha de género → ODS 5 con 63 % de puntuación relativa) y muestra ranking
top-3. Se corrigió la UI para llamar "puntuación relativa" (no
"probabilidad") al softmax de márgenes de LinearSVC.

## 9. Pendientes / decisiones que corresponden al estudiante

- **Autor del notebook**: la cabecera mantiene el nombre que traía el
  esqueleto ("Benjamin Ricardo Collante Juvinao"); el usuario de git es
  "Sara Gallego". Ajustar autor(es) antes de entregar.
- Revisar y, si se desea, reescribir con voz propia las interpretaciones de
  tópicos (§7) y las conclusiones (§12): están redactadas a partir de los
  resultados reales pero son la parte que la rúbrica evalúa como razonamiento
  del estudiante.
- Opcional: ampliar el grid a 400–500 componentes (la tendencia sigue
  creciendo) y calibrar probabilidades (`CalibratedClassifierCV`) para la app.
- `git add` de los nuevos archivos y commit (no se ha hecho ningún commit).

## 10. Ajustes de formato en el notebook (2026-09-19)

- Se probó un diagrama Mermaid para la arquitectura de la sección 8; el
  entorno de notebook del estudiante no lo renderiza, así que se **revirtió**
  al bloque de texto plano original (con flechas `->` / `<-`).
- Se normalizó la tipografía en todas las celdas (markdown y código):
  guiones largos (`—`, `–`) sustituidos por `-` (rangos numéricos, separadores)
  o `:` (introducción de una explicación), y flechas Unicode (`→`, `←`, `↔`)
  por `->`, `<-`, `<->`. Motivo: evitar caracteres que algunos editores/
  exportadores muestran mal y facilitar la edición.
- Como el cambio tocó cadenas de `print` y títulos de figuras, el notebook se
  reejecutó completo con `scripts/run_notebook.py` y se regeneraron los
  entregables. Resultados numéricos idénticos (misma semilla).

## 11. Rediseño visual de la app Streamlit (2026-09-19)

Mejora puramente de presentación del bono (criterio "app Streamlit
funcional"); no se tocó `src/` ni la lógica de predicción (`predict`,
`load_pipeline`, `clean_text` siguen igual).

- **Tema** (`.streamlit/config.toml`, nuevo): paleta clara con acento verde
  azulado (`#0F766E`), tipografía Manrope (títulos) + Inter (cuerpo) vía
  Google Fonts.
- **Colores oficiales ONU por ODS** (`ODS_COLORS` en `app/streamlit_app.py`):
  se usan como acento visual (franja del resultado, chips de afinidad, barras
  del ranking, leyenda de clases en la sidebar) para que el color siempre
  vaya acompañado del número y nombre del ODS, nunca como único portador de
  información.
- Encabezado ("hero") con el objetivo del proyecto y una tira visual del
  pipeline (TF-IDF → LSA (SVD) → Clasificador).
- Selector de ejemplos con `st.pills` (antes `st.selectbox`); tarjeta de
  resultado destacada con el color oficial del ODS predicho; top-3 ODS más
  afines como tarjetas con barra de progreso; ranking completo de los 16 ODS
  como gráfico de barras horizontal coloreado (Altair, con tooltip) en vez de
  `st.bar_chart`.
- Sidebar reorganizada: métricas nativas de Streamlit para componentes SVD y
  número de clases, y chips de color por ODS como leyenda.
- **Bug encontrado y corregido durante la prueba en navegador**: un `<div
  class="card">` abierto con `st.markdown` y cerrado más abajo no envuelve
  widgets nativos (`st.pills`, `st.text_area`, `st.button`) porque cada
  llamada de Streamlit se renderiza en su propio bloque del DOM; el resultado
  era una caja vacía antes del contenido real. Se reemplazó por
  `st.container(border=True)`, que sí es un contenedor real.
- Verificado end-to-end en navegador (Chrome, vía herramienta de automatización)
  con el pipeline real (`SVD 300, LinearSVC, clases 1-16`): ejemplo de
  educación → ODS 4 con 65 % de puntuación relativa, chips y gráfico
  coloreados correctamente, tabla de puntuaciones y texto normalizado
  visibles.

## 11. Entregables autocontenidos: solo notebook + `streamlit_app.py` (2026-09-19)

**Restricción nueva:** en la plataforma solo se pueden adjuntar dos archivos,
el notebook y la app. Ninguno puede depender del paquete `src/`.

**Qué se hizo:**

- **Notebook**: el código de `src/` se movió a celdas dentro de la sección
  donde se usa (limpieza/stopwords/TF-IDF en §6, LSA en §7, pipeline y grid en
  §8, métricas en §10, tabla de evidencia en §11). El catálogo `ODS_NAMES` y
  `load_dataset` están en §2-§3. El notebook busca el Excel en `data/`,
  `../data/` o junto a él (`Datos_textosODS.xlsx` / `textos_ods.xlsx`).
  Verificado: 0 referencias a `src`, sintaxis válida en todas las celdas.
- **App**: se conservó el rediseño de la interfaz (colores ONU, Altair, CSS)
  y se hizo autocontenida: define `clean_text` y `ODS_NAMES`; busca
  `pipeline_ods.joblib` junto a la app, en `models/`, `../models/` o el cwd, y
  si no lo encuentra permite escribir la ruta o **subir el archivo** desde la
  barra lateral.
- **Detalle técnico clave**: el `TfidfVectorizer` serializado referencia la
  función de limpieza por nombre calificado (`__main__.clean_text` al entrenar
  en el notebook). La app registra `clean_text` en `sys.modules["__main__"]` y
  en un módulo shim `src.preprocessing` antes de `joblib.load`, por lo que
  puede cargar modelos entrenados con esta versión o con la anterior.
- `src/` se eliminó del repo para que no haya lógica duplicada que pueda
  divergir; `clean_text`/`ODS_NAMES` están duplicados a propósito (y solo ahí)
  entre notebook y app. `scripts/run_notebook.py` copia ahora también la app a
  `entregables/`.
- `CLAUDE.md`, `README.md` y `CHECKLIST.md` actualizados.
