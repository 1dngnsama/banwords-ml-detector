import re
from typing import Dict, List

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tektur:wght@400;500;600;700;800;900&display=swap');
    html, body, body * { font-family: "Tektur", sans-serif !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

LABEL_COLUMNS: List[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    stop_words="english",
                    max_features=100_000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=200,
                        class_weight="balanced",
                        n_jobs=-1,
                        solver="liblinear",
                    ),
                    n_jobs=-1,
                ),
            ),
        ]
    )


@st.cache_data(show_spinner=False)
def load_data(sample_size: int = 20_000) -> pd.DataFrame:
    df = pd.read_csv("train.csv")
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    return df[["comment_text", *LABEL_COLUMNS]]


@st.cache_resource(show_spinner=False)
def train_model(training_data: pd.DataFrame) -> Pipeline:
    X_train, X_test, y_train, _ = train_test_split(
        training_data["comment_text"],
        training_data[LABEL_COLUMNS],
        test_size=0.2,
        random_state=42,
        stratify=training_data[LABEL_COLUMNS].sum(axis=1) > 0,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_labels(model: Pipeline, text: str) -> Dict[str, float]:
    probabilities = model.predict_proba([text])[0]
    return {label: prob for label, prob in zip(LABEL_COLUMNS, probabilities)}


st.set_page_config(page_title="Toxic classificator", page_icon='👾')
st.title("NLP - Класифікатор текстів")
st.subheader(
    "Введіть коментар нижче, щоб класифікувати його!", anchor=False
    
)

with st.spinner("Завантаження даних..."):
    data = load_data()

with st.spinner("Тренування моделі..."):
    model = train_model(data)

with st.form("comment_form"):
    user_text = st.text_area("Коментар", height=200)
    threshold = st.slider(
        "Поріг класифікації", 0.0, 1.0, 0.5, 0.05, help="Значення вище порогу вважаються токсичними"
    )
    submitted = st.form_submit_button("Оцінити")

if submitted:
    if not user_text.strip():
        st.warning("Будь ласка, введіть текст для аналізу.")
    else:
        with st.spinner("Отримання прогнозу..."):
            probabilities = predict_labels(model, user_text)

        st.subheader("Ймовірності класів")
        st.write(
            """
            Таблиця показує ймовірність належності коментаря до кожної категорії. 
            Значення вище обраного порогу ввожатимемо токсичними.
            """
        )

        results = [
            {
                "Категорія": label,
                "Ймовірність": f"{prob:.2%}",
                "Негативний": "✅" if prob >= threshold else "❌",
            }
            for label, prob in probabilities.items()
        ]

        st.table(results)

        toxic_labels = [label for label, prob in probabilities.items() if prob >= threshold]
        if toxic_labels:
            st.error(
                "Коментар вважається токсичним для категорій: "
                + ", ".join(toxic_labels)
            )
        else:
            st.success("Ознак токсичності не виявлено за заданим порогом.")

c1,c2,c3 = st.columns([2,6,2])
with c2:
    st.caption("ДЕРЖАВНИЙ ТОРГОВЕЛЬНО-ЕКОНОМІЧНИЙ УНІВЕРСИТЕТ")
    col_1, col_2, col_3 = st.columns([2,4,2])
    with col_2:
        st.caption("КИЇВ - 2025")