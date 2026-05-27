from pathlib import Path
import joblib
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data_utils import clean_text

BOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = BOT_DIR / "saved_models"


def load_all_models():

    models = {}

    models["naive_bayes_model"] = joblib.load(MODELS_DIR / "naive_bayes_model.pkl")
    models["naive_bayes_vectorizer"] = joblib.load(MODELS_DIR / "naive_bayes_vectorizer.pkl")

    models["logreg_model"] = joblib.load(MODELS_DIR / "logreg_model.pkl")
    models["logreg_vectorizer"] = joblib.load(MODELS_DIR / "logreg_vectorizer.pkl")

    models["textcnn_model"] = load_model(MODELS_DIR / "textcnn_model.keras")
    models["textcnn_tokenizer"] = joblib.load(MODELS_DIR / "textcnn_tokenizer.pkl")
    models["textcnn_label_encoder"] = joblib.load(MODELS_DIR / "textcnn_label_encoder.pkl")
    models["textcnn_max_len"] = joblib.load(MODELS_DIR / "textcnn_max_len.pkl")

    return models


MODELS = None


def get_models():

    global MODELS

    if MODELS is None:
        MODELS = load_all_models()

    return MODELS


def predict_sklearn_model(model, vectorizer, text):

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]
    confidence = float(np.max(probabilities))

    return prediction, confidence


def predict_textcnn(model, tokenizer, label_encoder, max_len, text):

    cleaned = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len, padding="post", truncating="post")

    probabilities = model.predict(padded, verbose=0)[0]

    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])

    prediction = label_encoder.inverse_transform([class_index])[0]

    return prediction, confidence


def predict_all(text):

    models = get_models()

    nb_prediction, nb_confidence = predict_sklearn_model(
        models["naive_bayes_model"], models["naive_bayes_vectorizer"], text
    )

    logreg_prediction, logreg_confidence = predict_sklearn_model(
        models["logreg_model"], models["logreg_vectorizer"], text
    )

    textcnn_prediction, textcnn_confidence = predict_textcnn(
        models["textcnn_model"],
        models["textcnn_tokenizer"],
        models["textcnn_label_encoder"],
        models["textcnn_max_len"],
        text,
    )

    return {
        "Naive Bayes": {"emotion": nb_prediction, "confidence": nb_confidence},
        "Logistic Regression": {
            "emotion": logreg_prediction,
            "confidence": logreg_confidence,
        },
        "TextCNN": {"emotion": textcnn_prediction, "confidence": textcnn_confidence},
    }


def format_predictions(predictions):

    lines = []

    for model_name, result in predictions.items():
        emotion = result["emotion"]
        confidence = result["confidence"] * 100

        lines.append(
            f"Предполагаемая эмоция модели {model_name}: {emotion}\n"
            f"Уверенность модели: {confidence:.0f}%"
        )

    return "\n\n".join(lines)


if __name__ == "__main__":
    text = input("Введите текст: ")

    predictions = predict_all(text)
    print(format_predictions(predictions))
