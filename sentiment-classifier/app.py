import os
import sys
import tempfile
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import (  # noqa: E402
    TEXT_COLUMN,
    default_output_path,
    find_latest_model,
    load_label_classes,
    predict_dataframe,
    predict_sentiment,
)


@st.cache_resource
def load_sentiment_model(model_path):
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    label_classes = load_label_classes(Path(model_path))
    return model, label_classes


def list_models():
    return sorted(Path("models").glob("sentiment_model_*.keras"), reverse=True)


def render_probabilities(label_classes, probabilities):
    probability_data = pd.DataFrame(
        {
            "sentiment": label_classes,
            "probability": [float(probability) for probability in probabilities],
        }
    )
    st.dataframe(
        probability_data.style.format({"probability": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(probability_data, x="sentiment", y="probability")


def render_text_mode(model_path, label_classes):
    review_text = st.text_area(
        "Text",
        height=180,
        placeholder="Write a review, comment or phrase to classify...",
    )

    if st.button("Predict Text", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("Write a text before predicting.")
            return

        with st.spinner("Loading model and generating prediction..."):
            model, label_classes = load_sentiment_model(str(model_path))
            prediction, probabilities = predict_sentiment(model, label_classes, review_text)

        st.subheader(f"Prediction: {prediction}")
        render_probabilities(label_classes, probabilities)


def render_database_mode(model_path, label_classes):
    st.caption(f"The CSV must have a column named `{TEXT_COLUMN}`.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    local_path = st.text_input(
        "Or use local path",
        placeholder="data/external/sample_reviews.csv",
    )

    if st.button("Predict Database", type="primary", use_container_width=True):
        input_path = None
        temp_file = None

        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            temp_file.write(uploaded_file.getbuffer())
            temp_file.close()
            input_path = Path(temp_file.name)
        elif local_path.strip():
            input_path = Path(local_path.strip())
        else:
            st.warning("Upload a CSV or write a local path.")
            return

        try:
            with st.spinner("Loading model and generating predictions..."):
                model, label_classes = load_sentiment_model(str(model_path))
                predictions = predict_dataframe(model, label_classes, input_path)
        except Exception as error:
            st.error(f"Could not generate prediction: {error}")
            return
        finally:
            if temp_file is not None:
                Path(temp_file.name).unlink(missing_ok=True)

        st.subheader("Results")
        st.dataframe(predictions, use_container_width=True, hide_index=True)

        csv_data = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if local_path.strip():
            output_path = default_output_path(Path(local_path.strip()))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(output_path, index=False)
            st.success(f"Predictions saved to {output_path}")


def main():
    st.set_page_config(page_title="Sentiment Classifier", page_icon=":chart_with_upwards_trend:")
    st.title("Sentiment Classifier")

    model_paths = list_models()
    if not model_paths:
        st.error("No models found in models/. Please train a model first.")
        return

    latest_model = find_latest_model()
    selected_model = st.sidebar.selectbox(
        "Model",
        options=model_paths,
        index=model_paths.index(latest_model),
        format_func=lambda path: path.name,
    )

    label_classes = load_label_classes(selected_model)
    st.sidebar.caption(f"Classes: {', '.join(label_classes)}")

    mode = st.radio(
        "Input",
        options=["Text", "Database"],
        horizontal=True,
    )

    if mode == "Text":
        render_text_mode(selected_model, label_classes)
    else:
        render_database_mode(selected_model, label_classes)


if __name__ == "__main__":
    main()
