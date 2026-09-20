"""Aplicación Streamlit del clasificador de textos ODS (bono del Microproyecto 2).

Archivo **autocontenido**: no depende de ningún módulo del repositorio. Carga
el pipeline entrenado en el notebook (``pipeline_ods.joblib``) y lo aplica tal
cual a un texto libre: la limpieza, TF-IDF, SVD y clasificador son exactamente
los mismos objetos ajustados durante el entrenamiento, por lo que el texto del
usuario recibe el mismo tratamiento que los datos de entrenamiento.

Ejecutar (desde cualquier carpeta)::

    python -m streamlit run streamlit_app.py

El modelo se busca automáticamente junto a este archivo, en ``models/`` o en
``../models/``; si no se encuentra, la app permite indicar la ruta o subir el
archivo ``.joblib``.
"""

from __future__ import annotations

import io
import re
import sys
import types
import unicodedata
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Catálogo de ODS y limpieza de texto (idénticos a los del notebook)
# --------------------------------------------------------------------------- #
ODS_NAMES: dict[int, str] = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_LETTER_RE = re.compile(r"[^a-záéíóúüñ\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text):
    """Normaliza un texto en español (misma función usada al entrenar el TF-IDF)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = _NON_LETTER_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def _register_clean_text() -> None:
    """Hace que ``clean_text`` sea resoluble al des-serializar el pipeline.

    El ``TfidfVectorizer`` guarda una referencia a la función de limpieza por su
    nombre calificado (``__main__.clean_text`` si se entrenó en el notebook, o
    ``src.preprocessing.clean_text`` en versiones anteriores). Se registran
    ambas rutas para que ``joblib.load`` funcione sin importar cómo se entrenó.
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "clean_text"):
        setattr(main_mod, "clean_text", clean_text)
    if "src.preprocessing" not in sys.modules:
        pkg = sys.modules.setdefault("src", types.ModuleType("src"))
        shim = types.ModuleType("src.preprocessing")
        shim.clean_text = clean_text
        pkg.preprocessing = shim
        sys.modules["src.preprocessing"] = shim


_register_clean_text()

# --------------------------------------------------------------------------- #
# Localización del modelo
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
MODEL_NAME = "pipeline_ods.joblib"
CANDIDATOS = [
    HERE / MODEL_NAME,
    HERE / "models" / MODEL_NAME,
    HERE.parent / "models" / MODEL_NAME,
    Path.cwd() / MODEL_NAME,
    Path.cwd() / "models" / MODEL_NAME,
]

# Colores oficiales de la ONU para cada ODS (identidad visual reconocible:
# https://www.un.org/sustainabledevelopment/news/communications-material/).
# Se usan solo como acento visual (chips, barras); nunca son el único medio
# para transmitir información: el número y el nombre siempre acompañan.
ODS_COLORS: dict[int, str] = {
    1: "#E5243B", 2: "#DDA63A", 3: "#4C9F38", 4: "#C5192D",
    5: "#FF3A21", 6: "#26BDE2", 7: "#FCC30B", 8: "#A21942",
    9: "#FD6925", 10: "#DD1367", 11: "#FD9D24", 12: "#BF8B2E",
    13: "#3F7E44", 14: "#0A97D9", 15: "#56C02B", 16: "#00689D",
    17: "#19486A",
}

EJEMPLOS = {
    "Educación": (
        "Los docentes de primaria reportan que la deserción escolar aumentó en las zonas rurales, "
        "y las escuelas carecen de materiales de aprendizaje y de formación para el profesorado."
    ),
    "Agua": (
        "La gestión de las aguas subterráneas y el tratamiento de aguas residuales son fundamentales "
        "para garantizar el acceso a agua potable en las comunidades más vulnerables."
    ),
    "Energía": (
        "La expansión de las energías renovables y la mejora de la eficiencia energética permiten "
        "reducir el costo de la electricidad y la dependencia de los combustibles fósiles."
    ),
    "Justicia": (
        "El derecho internacional de los derechos humanos exige que los Estados garanticen el acceso "
        "a la justicia y el debido proceso ante tribunales independientes."
    ),
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .hero-title { font-family: 'Manrope', sans-serif; }

#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 880px; }

/* ---- Hero ---- */
.hero {
    background: linear-gradient(135deg, #0F766E 0%, #115E59 100%);
    border-radius: 18px;
    padding: 2rem 2.25rem;
    margin-bottom: 1.75rem;
    color: #F0FDFA;
    box-shadow: 0 8px 24px rgba(15, 118, 110, 0.25);
}
.hero-eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.72rem;
    font-weight: 600;
    color: #99F6E4;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
    color: #FFFFFF;
    line-height: 1.25;
}
.hero-sub { font-size: 0.95rem; color: #CCFBF1; margin: 0; line-height: 1.5; }

/* ---- Pipeline strip ---- */
.pipeline-strip { display: flex; align-items: center; gap: 0.5rem; margin-top: 1.1rem; flex-wrap: wrap; }
.pipeline-step {
    display: flex; align-items: center; gap: 0.45rem;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 999px;
    padding: 0.3rem 0.75rem 0.3rem 0.45rem;
    font-size: 0.8rem; font-weight: 600; color: #F0FDFA;
}
.pipeline-step .dot {
    width: 1.3rem; height: 1.3rem; border-radius: 50%;
    background: #FFFFFF; color: #0F766E;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 800;
}
.pipeline-arrow { color: #5EEAD4; font-weight: 700; }

/* ---- Result ---- */
.result-card {
    display: flex; align-items: center; gap: 1rem;
    border-radius: 14px; padding: 1.1rem 1.4rem; margin: 0.25rem 0 1.25rem 0;
    color: #FFFFFF; box-shadow: 0 6px 18px rgba(0,0,0,0.12);
}
.result-badge {
    flex-shrink: 0; width: 3rem; height: 3rem; border-radius: 12px;
    background: rgba(255,255,255,0.22); display: flex; align-items: center;
    justify-content: center; font-size: 1.3rem; font-weight: 800;
}
.result-text .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.85; }
.result-text .name { font-size: 1.15rem; font-weight: 700; margin-top: 0.1rem; }

/* ---- Affinity chips ---- */
.chip {
    border-radius: 12px; padding: 0.7rem 0.85rem; border: 1px solid #E2E8F0;
    border-left: 5px solid var(--chip-color, #0F766E); background: #FFFFFF;
    height: 100%;
}
.chip .chip-top { display: flex; justify-content: space-between; align-items: baseline; }
.chip .chip-ods { font-size: 0.78rem; font-weight: 700; color: #475569; }
.chip .chip-score { font-size: 1.05rem; font-weight: 800; color: #0F172A; }
.chip .chip-name { font-size: 0.82rem; color: #334155; margin-top: 0.2rem; }
.chip .chip-bar-track { background: #F1F5F9; border-radius: 999px; height: 6px; margin-top: 0.55rem; overflow: hidden; }
.chip .chip-bar-fill { height: 100%; border-radius: 999px; background: var(--chip-color, #0F766E); }

/* ---- Legend chips (sidebar) ---- */
.legend-grid { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.4rem; }
.legend-item {
    display: flex; align-items: center; gap: 0.35rem;
    background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 999px;
    padding: 0.18rem 0.6rem 0.18rem 0.32rem; font-size: 0.74rem; color: #334155;
}
.legend-dot { width: 0.6rem; height: 0.6rem; border-radius: 50%; background: var(--dot-color, #0F766E); }

/* Buttons */
.stButton > button {
    border-radius: 10px; font-weight: 600; border: none;
    background: #0F766E; color: #FFFFFF;
}
.stButton > button:hover { background: #115E59; color: #FFFFFF; }
</style>
"""


@st.cache_resource(show_spinner="Cargando el pipeline entrenado…")
def load_pipeline_from_path(path: str):
    return joblib.load(path)


@st.cache_resource(show_spinner="Cargando el pipeline subido…")
def load_pipeline_from_bytes(data: bytes):
    return joblib.load(io.BytesIO(data))


def find_model_path() -> Path | None:
    return next((p for p in CANDIDATOS if p.exists()), None)


def get_pipeline():
    """Obtiene el pipeline: ruta automática, ruta escrita por el usuario o archivo subido."""
    auto = find_model_path()
    if auto is not None:
        return load_pipeline_from_path(str(auto)), auto
    with st.sidebar:
        st.markdown("### Modelo")
        st.warning(
            f"No se encontró `{MODEL_NAME}` junto a la app ni en `models/`. "
            "Ejecuta el notebook para generarlo, indica su ruta o súbelo aquí."
        )
        ruta = st.text_input("Ruta al archivo .joblib", value="")
        if ruta and Path(ruta).exists():
            return load_pipeline_from_path(ruta), Path(ruta)
        subido = st.file_uploader("…o sube el archivo pipeline_ods.joblib", type=["joblib", "pkl"])
        if subido is not None:
            return load_pipeline_from_bytes(subido.getvalue()), Path(subido.name)
    return None, None


def predict(pipeline, text: str) -> tuple[int, pd.DataFrame | None, bool]:
    """Devuelve el ODS predicho, la tabla de puntuaciones por clase y si son probabilidades reales.

    Con ``predict_proba`` (p. ej. LogisticRegression) la columna ``score`` son probabilidades.
    Con ``decision_function`` (LinearSVC) se aplica un softmax a los márgenes solo para obtener un
    ranking comparable entre clases; **no** es una probabilidad calibrada y la UI lo indica.
    """
    pred = int(pipeline.predict([text])[0])
    table, es_proba = None, False
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        scores = pipeline.predict_proba([text])[0]
        es_proba = True
    elif hasattr(clf, "decision_function"):
        margins = pipeline.decision_function([text])[0]
        e = np.exp(margins - margins.max())
        scores = e / e.sum()
    else:
        return pred, None, False
    table = pd.DataFrame({"ODS": clf.classes_.astype(int), "score": scores})
    table["nombre"] = table["ODS"].map(ODS_NAMES)
    table["color"] = table["ODS"].map(ODS_COLORS)
    table = table.sort_values("score", ascending=False).reset_index(drop=True)
    return pred, table, es_proba


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Microproyecto 2 · Machine Learning No Supervisado</div>
            <div class="hero-title">🌍 Clasificador de textos según los ODS</div>
            <p class="hero-sub">
                Identifica a qué Objetivo de Desarrollo Sostenible de la ONU pertenece un texto en
                español, entrenado con textos etiquetados de los ODS 1 a 16 (el ODS 17 no está en los datos).
            </p>
            <div class="pipeline-strip">
                <div class="pipeline-step"><span class="dot">1</span> TF-IDF</div>
                <span class="pipeline-arrow">-></span>
                <div class="pipeline-step"><span class="dot">2</span> LSA (SVD)</div>
                <span class="pipeline-arrow">-></span>
                <div class="pipeline-step"><span class="dot">3</span> Clasificador</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(pipeline, model_path: Path | None) -> None:
    with st.sidebar:
        st.markdown("### Acerca del modelo")
        if model_path is not None:
            st.caption(f"Archivo: `{model_path.name}`")
        steps = " -> ".join(name for name, _ in pipeline.steps)
        st.markdown(f"**Pipeline:** `{steps}`")

        svd = pipeline.named_steps.get("svd")
        clf = pipeline.named_steps["clf"]
        col1, col2 = st.columns(2)
        if svd is not None:
            col1.metric("Componentes SVD", svd.n_components)
        col2.metric("Clases ODS", len(clf.classes_))
        st.markdown(f"**Clasificador:** `{type(clf).__name__}`")

        st.markdown("**Clases conocidas**")
        chips = "".join(
            f'<span class="legend-item" style="--dot-color:{ODS_COLORS.get(int(c), "#0F766E")}">'
            f'<span class="legend-dot"></span>ODS {int(c)}</span>'
            for c in sorted(clf.classes_)
        )
        st.markdown(f'<div class="legend-grid">{chips}</div>', unsafe_allow_html=True)

        st.divider()
        st.caption(
            "El texto ingresado pasa por la **misma** limpieza, vectorización y reducción "
            "de dimensionalidad que los datos de entrenamiento (la limpieza va dentro del pipeline)."
        )


def render_ranking_chart(table: pd.DataFrame, etiqueta: str) -> None:
    chart_df = table.copy()
    chart_df["label"] = chart_df.apply(lambda r: f"ODS {int(r['ODS'])} · {r['nombre']}", axis=1)
    domain = [int(o) for o in sorted(chart_df["ODS"].unique())]
    range_ = [ODS_COLORS[o] for o in domain]

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("score:Q", title=etiqueta, axis=alt.Axis(format="%")),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.Color(
                "ODS:N",
                scale=alt.Scale(domain=domain, range=range_),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("label:N", title="ODS"),
                alt.Tooltip("score:Q", title=etiqueta, format=".1%"),
            ],
        )
        .properties(height=max(220, 26 * len(chart_df)))
    )
    st.altair_chart(chart, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Clasificador ODS", page_icon="🌍", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    render_hero()
    pipeline, model_path = get_pipeline()
    if pipeline is None:
        st.info("Carga el modelo desde la barra lateral para empezar.")
        st.stop()

    render_sidebar(pipeline, model_path)

    with st.container(border=True):
        st.markdown("**Ejemplo rápido** _(opcional)_")
        ejemplo = st.pills(
            "Ejemplo rápido",
            options=list(EJEMPLOS.keys()),
            selection_mode="single",
            label_visibility="collapsed",
        )
        texto = st.text_area(
            "Texto a clasificar",
            value=EJEMPLOS.get(ejemplo, ""),
            height=170,
            placeholder="Pega aquí un párrafo en español (idealmente 50-200 palabras)…",
            label_visibility="collapsed",
        )
        clasificar = st.button("Clasificar texto", type="primary", width="stretch")

    if clasificar:
        limpio = clean_text(texto)
        if len(limpio.split()) < 5:
            st.warning("Ingresa un texto con al menos 5 palabras significativas.")
            st.stop()

        pred, table, es_proba = predict(pipeline, texto)
        color = ODS_COLORS.get(pred, "#0F766E")

        st.markdown(
            f"""
            <div class="result-card" style="background:{color};">
                <div class="result-badge">{pred}</div>
                <div class="result-text">
                    <div class="label">ODS predicho</div>
                    <div class="name">{ODS_NAMES[pred]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if table is not None:
            etiqueta = "Probabilidad" if es_proba else "Puntuación relativa"
            st.caption(
                f"{etiqueta} de los 3 ODS más afines. "
                + ("" if es_proba else "El clasificador (LinearSVC) no produce probabilidades: la puntuación es "
                   "un softmax de los márgenes de decisión, útil solo como ranking.")
            )

            top = table.head(3)
            cols = st.columns(3)
            max_score = float(table["score"].max())
            for col, (_, row) in zip(cols, top.iterrows()):
                pct = row["score"] / max_score * 100 if max_score else 0
                col.markdown(
                    f"""
                    <div class="chip" style="--chip-color:{row['color']}">
                        <div class="chip-top">
                            <span class="chip-ods">ODS {int(row['ODS'])}</span>
                            <span class="chip-score">{row['score']:.0%}</span>
                        </div>
                        <div class="chip-name">{row['nombre']}</div>
                        <div class="chip-bar-track"><div class="chip-bar-fill" style="width:{pct:.0f}%"></div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander(f"Ver {etiqueta.lower()} para todos los ODS"):
                render_ranking_chart(table, etiqueta)
                st.dataframe(
                    table[["ODS", "nombre", "score"]]
                    .rename(columns={"score": etiqueta.lower(), "nombre": "ODS (nombre)"})
                    .style.format({etiqueta.lower(): "{:.3f}"}),
                    hide_index=True,
                    width="stretch",
                )

        with st.expander("Ver texto normalizado (entrada real al TF-IDF)"):
            st.code(limpio, language=None)


if __name__ == "__main__":
    main()
