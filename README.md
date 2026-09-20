# Microproyecto 2 — Clasificación de textos según los ODS

Solución de NLP + machine learning que clasifica textos en español según los
Objetivos de Desarrollo Sostenible (ODS) de la ONU, desarrollada para el curso
*Machine Learning No Supervisado* de la Maestría en Inteligencia Artificial.

**Autores:** Benjamin Ricardo Collante Juvinao, Sara Gallego Villada

## Tabla de contenido

- [Contexto y objetivo](#contexto-y-objetivo)
- [Dataset](#dataset)
- [Enfoque técnico](#enfoque-técnico)
  - [1. Limpieza y vectorización TF-IDF](#1-limpieza-y-vectorización-tf-idf)
  - [2. LSA — análisis semántico latente](#2-lsa--análisis-semántico-latente)
  - [3. Clasificador con reducción de dimensionalidad](#3-clasificador-con-reducción-de-dimensionalidad)
  - [4. Evaluación](#4-evaluación)
- [Resultados](#resultados)
- [App Streamlit (bono)](#app-streamlit-bono)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Cómo reproducir](#cómo-reproducir)
- [Entregables](#entregables-solo-se-adjuntan-dos-archivos)

## Contexto y objetivo

La Agenda 2030 de la ONU define 17 Objetivos de Desarrollo Sostenible (ODS).
Organismos como el UNFPA analizan grandes volúmenes de texto (planes de
desarrollo, aportes de participación ciudadana, reportes) y necesitan
relacionar cada documento con el ODS que le corresponde. El objetivo del
proyecto es automatizar esa clasificación con un pipeline reproducible de
principio a fin: nada de pasos manuales entre el texto crudo y la predicción.

El enunciado exige cuatro actividades, todas resueltas dentro de un único
notebook (`notebooks/microproyecto2_ods.ipynb`):

| Actividad | Sección del notebook |
|---|---|
| 1. Bolsa de palabras + TF-IDF dentro de un pipeline | 6 |
| 2. LSA con `TruncatedSVD` (10-20 componentes), ≥5 tópicos interpretados frente a los ODS | 7 |
| 3. Clasificador con reducción de dimensionalidad y búsqueda de hiperparámetros, justificado | 8-9 |
| 4. Evaluación sobre test no visto, ≥4 textos mostrados | 10-11 |
| Bono: app Streamlit que reutiliza el mismo pipeline | 13 y `app/streamlit_app.py` |

## Dataset

`data/Datos_textosODS.xlsx` — 9 656 filas, columnas `textos` (str) y `ODS`
(int). Verificado en la auditoría inicial:

- Sin nulos ni duplicados.
- Solo aparecen las etiquetas **1 a 16**; el ODS 17 no está representado en
  los datos (por eso el modelo nunca lo predice, aunque el catálogo
  `ODS_NAMES` incluya su nombre para que la UI de la app sea completa).
- **Desbalance moderado**: de 312 textos (ODS 12, el menos representado) a
  1 080 (ODS 16, el más representado) — razón ~3.5:1. Esto es lo que hace que
  la métrica principal del proyecto sea **F1 macro** en vez de accuracy, y que
  el clasificador use `class_weight="balanced"`.
- Textos de ~110 palabras en promedio (709 caracteres), traducidos con DeepL
  y, en algunos casos, aumentados con ChatGPT.

El split se hace una sola vez, antes de tocar TF-IDF/SVD/modelo:

```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

7 724 textos para entrenamiento/validación cruzada, 1 932 para el test final.
El test **no se usa nunca** para ajustar el vectorizador, elegir el modelo o
buscar hiperparámetros — solo se evalúa una vez, al final (sección 10).

## Enfoque técnico

Todo el pipeline vive dentro de objetos `sklearn.pipeline.Pipeline`, de modo
que TF-IDF, SVD y el clasificador se reajustan en cada fold de la validación
cruzada (sin fuga de información) y el objeto final se puede serializar tal
cual con `joblib` y reutilizar desde la app Streamlit.

### 1. Limpieza y vectorización TF-IDF

`clean_text()` normaliza cada texto: minúsculas, elimina URLs, normaliza
Unicode a NFC, quita dígitos y puntuación **conservando tildes y ñ**, colapsa
espacios. No se aplica *stemming* ni lematización a propósito: el corpus es
formal y ya está homogéneamente traducido, así que la ganancia de un stemmer
sería marginal frente a la pérdida de interpretabilidad de los términos que
luego aparecen en los tópicos LSA y en los coeficientes del clasificador.

Esa función se inyecta como `preprocessor` del propio `TfidfVectorizer`, así
que la limpieza vive *dentro* del pipeline y no como paso externo:

```python
TfidfVectorizer(
    preprocessor=clean_text,
    stop_words=get_spanish_stopwords(),   # NLTK + lista de dominio muy frecuente/poco discriminante
    ngram_range=(1, 2),                   # unigramas + bigramas
    min_df=3,                             # descarta ruido/errores de OCR-traducción
    max_df=0.90,                          # descarta términos casi universales
    sublinear_tf=True,                    # 1 + log(tf): repetir una palabra no la hace n veces más importante
    max_features=30_000,
)
```

Cada parámetro está justificado con una tabla en la sección 6 del notebook.

### 2. LSA — análisis semántico latente

`TruncatedSVD` se usa de **dos formas distintas** en el proyecto, y el
notebook lo explica de forma explícita para no mezclarlas:

- **LSA interpretativo (sección 7, 15 componentes)**: el objetivo aquí no es
  maximizar desempeño sino **interpretar tópicos**. Se comparó k=10/15/20 y se
  fijó 15 (dentro del rango 10-20 exigido). Cada componente se lee por sus
  términos de mayor peso absoluto (una componente LSA contrasta dos polos,
  positivo y negativo) y se contrasta cuantitativamente contra el ODS real
  calculando el puntaje medio de la componente por clase. Con eso se
  identificaron más de 12 componentes interpretables, muy por encima del
  mínimo de 5 exigido por la rúbrica. Ejemplos:

  | Componente | Términos principales | ODS dominante (score medio) |
  |---|---|---|
  | C2 | derechos, derechos humanos, internacional, artículo, ley, justicia | ODS 16 — Paz, justicia e instituciones sólidas (0.129) |
  | C5 | mujeres, género, hombres, igualdad, participación | ODS 5 — Igualdad de género (0.069) |
  | C7 | agua, aguas subterráneas, aguas residuales, agua potable | ODS 6 — Agua limpia y saneamiento (0.112) |
  | C11 | transporte, ciudades, rurales, zonas, público | ODS 11 — Ciudades sostenibles (0.042) / ODS 9 |

  Con solo 15 componentes la varianza explicada acumulada es baja (~3%),
  normal en matrices TF-IDF muy dispersas de vocabulario amplio: lo que
  importa para esta actividad es la interpretabilidad de los ejes, no cuánta
  varianza capturan.

- **SVD dentro del clasificador (secciones 8-9, 100/200/300 componentes)**:
  aquí sí es una reducción de dimensionalidad orientada a desempeño, buscada
  como hiperparámetro dentro del `GridSearchCV`. Como referencia cuantitativa,
  el notebook muestra que con solo 15 componentes (el LSA interpretativo) el
  F1 macro en CV cae a 0.715 — insuficiente para separar 16 clases — mientras
  que con 300 componentes sube a 0.849, casi igualando el 0.870 de no reducir
  dimensionalidad en absoluto. Esa comparación es la justificación explícita
  de por qué se necesitan más componentes para clasificar que para
  interpretar.

### 3. Clasificador con reducción de dimensionalidad

Arquitectura del pipeline final:

```
texto crudo -> TfidfVectorizer(preprocessor=clean_text, stopwords, 1-2 gramas, TF-IDF)
            -> TruncatedSVD(n_components=k)      <- reducción de dimensionalidad
            -> Normalizer(L2)                    <- paso clásico de LSA: cada documento a norma 1
            -> LogisticRegression | LinearSVC (class_weight="balanced")
```

El `Normalizer` L2 después del SVD es el paso clásico de LSA: hace que la
similitud entre documentos en el espacio latente dependa de la dirección
(temática) y no de la magnitud (longitud del texto).

**Búsqueda de hiperparámetros** con `GridSearchCV`, `StratifiedKFold(5,
shuffle=True, random_state=42)`, `refit="f1_macro"`, sobre **train
únicamente**:

| Modelo | `svd__n_components` | `C` |
|---|---|---|
| `LogisticRegression` | {100, 200, 300} | {1.0, 10.0, 30.0} |
| `LinearSVC` | {100, 200, 300} | {0.3, 1.0, 3.0} |

2 modelos × 3 componentes × 3 valores de `C` × 5 folds = **90 ajustes**.

Justificación del algoritmo: se compararon modelos lineales (LogReg,
LinearSVC) porque en espacios TF-IDF/LSA de alta dimensión con clases
linealmente bastante separables suelen igualar o superar a modelos no
lineales, con costo de entrenamiento muchísimo menor. `class_weight="balanced"`
compensa el desbalance moderado (312 a 1 080 textos por clase) sin necesitar
re-muestreo. Resultado de la búsqueda:

- **Ganador: `LinearSVC(C=0.3)` con 300 componentes SVD** — F1 macro CV
  0.849 ± 0.006, ~2× más rápido de entrenar que el mejor LogReg (F1 macro
  0.844). La tendencia es monótona: más componentes SVD (100→200→300) siempre
  mejora, y gana el `C` más pequeño (mayor regularización) en ambos modelos.
- Al no exponer `predict_proba`, el ranking top-3 de la app se calcula sobre
  `decision_function` (softmax de márgenes) y se etiqueta explícitamente como
  "puntuación relativa", nunca como probabilidad.

### 4. Evaluación

Una sola evaluación, al final, sobre los 1 932 textos de test nunca vistos
por ningún paso del ajuste (sección 10):

- Métricas globales: accuracy, precision/recall/F1 macro, F1 ponderado.
- Reporte por clase (precision/recall/F1 y soporte de las 16 clases).
- Matriz de confusión normalizada por fila + tabla de pares más confundidos.
- Evidencia sobre texto no visto (sección 11): 8 textos aleatorios del test +
  5 errores "seguros", mostrando fragmento, ODS real, ODS predicho, acierto y
  top-3 por margen — más del mínimo de 4 textos exigido por la rúbrica.
  Sección 13 añade 4 textos completamente nuevos, escritos a mano, como
  prueba adicional de generalización.

## Resultados

Última ejecución completa (`scripts/run_notebook.py`, notebook con kernel
reiniciado, sin errores). Estos son números reales de esa corrida, no metas:

| Métrica (test, 1 932 textos nunca vistos) | Valor |
|---|---|
| Accuracy | 0.885 |
| Precision macro | 0.853 |
| Recall macro | 0.855 |
| **F1 macro** | **0.854** |
| F1 ponderado | 0.885 |

- Validación cruzada (train, 5 folds): F1 macro 0.849 ± 0.006, accuracy
  0.874 — muy cerca del resultado en test, es decir, **no hay sobreajuste**
  al procedimiento de selección de modelo.
- Modelo final: `LinearSVC(C=0.3, class_weight="balanced")` sobre 300
  componentes de `TruncatedSVD`.
- Mejores clases (F1 > 0.9): ODS 16 — Paz, justicia e instituciones sólidas
  (0.97), ODS 4 — Educación de calidad (0.96), ODS 5 — Igualdad de género
  (0.95), ODS 3 — Salud y bienestar (0.94), ODS 6 — Agua limpia (0.94), ODS 7
  — Energía asequible (0.93), ODS 14 — Vida submarina (0.91).
- Clases más difíciles: ODS 8 — Trabajo decente (F1 0.59), ODS 10 —
  Reducción de las desigualdades (0.68), ODS 9 — Industria e innovación
  (0.74). La matriz de confusión muestra que estos tres ODS —el bloque
  temático trabajo/desigualdad/industria— se confunden entre sí con más
  frecuencia que con el resto, lo cual es consistente con que comparten
  vocabulario económico-social.

Detalle completo (por clase, matriz de confusión, pares confundidos,
interpretación) en la sección 10 del notebook ejecutado
(`entregables/microproyecto2_ods.html`) y en `docs/bitacora_desarrollo.md` §4.

## App Streamlit (bono)

`app/streamlit_app.py` es una app **autocontenida** (define su propia copia
de `clean_text` y `ODS_NAMES`, no importa nada del notebook) que carga
`models/pipeline_ods.joblib` — el mismo objeto `Pipeline` entrenado y
serializado en la sección 13 del notebook, sin ningún paso manual adicional —
y expone:

- Entrada de texto libre o selección de ejemplos por tema (educación, agua,
  energía, justicia, etc.) mediante `st.pills`.
- Predicción con el ODS ganador destacado con su color oficial ONU
  (`ODS_COLORS`), más un top-3 de ODS afines como tarjetas con barra de
  progreso.
- Ranking completo de los 16 ODS conocidos como gráfico de barras horizontal
  interactivo (Altair), y tabla con todas las puntuaciones.
- Expander con el texto ya normalizado, para ver exactamente qué entra al
  TF-IDF.
- Sidebar con la arquitectura del pipeline (TF-IDF → SVD → clasificador),
  número de componentes, y aviso explícito de que el ODS 17 no está entre las
  clases que el modelo puede predecir.
- Si no encuentra `pipeline_ods.joblib` junto a ella, en `models/`,
  `../models/` o el directorio actual, permite subirlo manualmente desde la
  barra lateral — así la app funciona sola en cualquier carpeta.

Detalle técnico: el `TfidfVectorizer` serializado referencia la función de
limpieza por nombre calificado (`__main__.clean_text`, tal como quedó al
entrenar en el notebook). La app registra `clean_text` en
`sys.modules["__main__"]` y en un módulo *shim* `src.preprocessing` antes de
`joblib.load`, para poder deserializar tanto modelos entrenados con esta
versión como con versiones anteriores que sí dependían de `src/`.

## Estructura del repositorio

```
data/Datos_textosODS.xlsx        dataset entregado (no se mueve ni se renombra)
notebooks/microproyecto2_ods.ipynb   notebook de trabajo, autocontenido (sin importar src/)
app/streamlit_app.py             app Streamlit (bono), autocontenida
entregables/                     microproyecto2_ods.{ipynb,html} + streamlit_app.py — lo que se entrega
models/pipeline_ods.joblib       pipeline final serializado (no versionado; lo genera el notebook)
outputs/figures, outputs/tables  salidas del notebook (no versionadas)
docs/bitacora_desarrollo.md      bitácora de desarrollo: qué faltaba, qué se hizo y por qué, etapa por etapa
scripts/run_notebook.py          ejecuta el notebook (evitando que el equipo se suspenda) y exporta entregables
CHECKLIST.md                     matriz de cumplimiento de la rúbrica con evidencia
references/                      enunciado del microproyecto y tutoriales del curso
```

`src/` se eliminó del repositorio: la plataforma de entrega solo admite dos
archivos adjuntos (notebook y `streamlit_app.py`), así que ambos son
autocontenidos y `clean_text`/`ODS_NAMES` están duplicados a propósito entre
ellos.

## Cómo reproducir

Requiere Python 3.14 (ver `requirements.txt`: scikit-learn, pandas, numpy,
nltk, matplotlib, seaborn, streamlit, joblib, openpyxl).

```powershell
.\setup.ps1                                          # crea .venv, instala requirements.txt y registra el kernel
.\.venv\Scripts\python.exe scripts\run_notebook.py   # ejecuta el notebook completo y copia notebook, HTML y app a entregables/ (~10 min)
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py   # app (usa models/pipeline_ods.joblib)
```

`scripts/run_notebook.py` corre `nbconvert --execute` con el kernel
reiniciado, impide que Windows suspenda el equipo durante el grid search de
hiperparámetros (que puede tardar 10-15 minutos), exporta el `.html` con
todas las salidas y copia `.ipynb`/`.html`/`streamlit_app.py` a
`entregables/`.

## Entregables (solo se adjuntan dos archivos)

- `entregables/microproyecto2_ods.ipynb` — notebook ejecutado de punta a
  punta, autocontenido (solo necesita el Excel del dataset junto a él o en
  `data/`).
- `entregables/streamlit_app.py` — app Streamlit autocontenida; necesita el
  `pipeline_ods.joblib` que genera el notebook (lo busca junto a ella o en
  `models/`, o se puede subir desde la barra lateral).
- `entregables/microproyecto2_ods.html` — exportación con todas las salidas
  visibles (por si se pide como respaldo).
