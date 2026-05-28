from pathlib import Path

import pytest

from bot.predict import format_predictions, predict_all

MODELS_DIR = Path("bot/saved_models")

REQUIRED_MODEL_FILES = [
    "naive_bayes_model.pkl",
    "naive_bayes_vectorizer.pkl",
    "logreg_model.pkl",
    "logreg_vectorizer.pkl",
    "textcnn_model.keras",
    "textcnn_tokenizer.pkl",
    "textcnn_label_encoder.pkl",
    "textcnn_max_len.pkl",
]


def test_full_prediction_pipeline():
    missing_files = [
        file_name
        for file_name in REQUIRED_MODEL_FILES
        if not (MODELS_DIR / file_name).exists()
    ]

    if missing_files:
        pytest.skip(
            "Модели ещё не обучены. " "Сначала запусти: python bot/train_models.py"
        )

    user_text = "Мне сегодня очень грустно и одиноко"

    predictions = predict_all(user_text)

    assert isinstance(predictions, dict)

    assert "Naive Bayes" in predictions
    assert "Logistic Regression" in predictions
    assert "TextCNN" in predictions

    allowed_emotions = {
        "joy",
        "sadness",
        "surprise",
        "fear",
        "anger",
    }

    for model_name, result in predictions.items():
        assert "emotion" in result
        assert "confidence" in result

        assert result["emotion"] in allowed_emotions
        assert 0 <= result["confidence"] <= 1

    answer = format_predictions(predictions)

    assert "Предполагаемая эмоция модели Naive Bayes" in answer
    assert "Предполагаемая эмоция модели Logistic Regression" in answer
    assert "Предполагаемая эмоция модели TextCNN" in answer
    assert "Уверенность модели" in answer
