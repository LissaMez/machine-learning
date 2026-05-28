from pathlib import Path
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data_utils import read_russian_emotion_file

BOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BOT_DIR.parent

DATA_DIR = PROJECT_DIR / "data" / "russian"
MODELS_DIR = BOT_DIR / "saved_models"

TRAIN_PATH = DATA_DIR / "ru-train.txt"
TEST_PATH = DATA_DIR / "ru-test.txt"


def train_naive_bayes(train_df, test_df):
    print("ОБУЧЕНИЕ NAIVE BAYES")

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)

    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_train = train_df["label"]
    y_test = test_df["label"]

    model = MultinomialNB(alpha=0.05)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 macro:", f1_score(y_test, y_pred, average="macro"))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODELS_DIR / "naive_bayes_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "naive_bayes_vectorizer.pkl")

    print("Naive Bayes сохранён.")


def train_logistic_regression(train_df, test_df):
    print("ОБУЧЕНИЕ LOGISTIC REGRESSION")

    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.9
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_train = train_df["label"]
    y_test = test_df["label"]

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 macro:", f1_score(y_test, y_pred, average="macro"))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODELS_DIR / "logreg_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "logreg_vectorizer.pkl")

    print("Logistic Regression сохранена.")


def train_textcnn(train_df, test_df):
    print("ОБУЧЕНИЕ TEXTCNN")

    max_words = 20000
    max_len = 60
    embedding_dim = 128
    filters = 128
    kernel_size = 5

    tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")

    tokenizer.fit_on_texts(train_df["text"])

    X_train_seq = tokenizer.texts_to_sequences(train_df["text"])
    X_test_seq = tokenizer.texts_to_sequences(test_df["text"])

    X_train = pad_sequences(
        X_train_seq, maxlen=max_len, padding="post", truncating="post"
    )
    X_test = pad_sequences(
        X_test_seq, maxlen=max_len, padding="post", truncating="post"
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"])
    y_test = label_encoder.transform(test_df["label"])

    num_classes = len(label_encoder.classes_)

    model = Sequential(
        [
            Embedding(
                input_dim=max_words, output_dim=embedding_dim, input_length=max_len
            ),
            Conv1D(filters=filters, kernel_size=kernel_size, activation="relu"),
            GlobalMaxPooling1D(),
            Dense(units=64, activation="relu"),
            Dropout(0.5),
            Dense(units=num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

    loss, accuracy = model.evaluate(X_test, y_test)
    print("TextCNN accuracy:", accuracy)

    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)

    print("F1 macro:", f1_score(y_test, y_pred, average="macro"))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    model.save(MODELS_DIR / "textcnn_model.keras")
    joblib.dump(tokenizer, MODELS_DIR / "textcnn_tokenizer.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "textcnn_label_encoder.pkl")
    joblib.dump(max_len, MODELS_DIR / "textcnn_max_len.pkl")

    print("TextCNN сохранена.")


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {TEST_PATH}")

    print("ЗАГРУЗКА РУССКИХ ДАННЫХ")

    train_df = read_russian_emotion_file(TRAIN_PATH, has_header=True)
    test_df = read_russian_emotion_file(TEST_PATH, has_header=False)

    print("Размер train:", len(train_df))
    print("Размер test:", len(test_df))

    if len(train_df) == 0:
        raise ValueError("Проверь ru-train.txt")

    if len(test_df) == 0:
        raise ValueError("Проверь ru-test.txt")

    print("\nКлассы в train:")
    print(train_df["label"].value_counts())

    print("\nКлассы в test:")
    print(test_df["label"].value_counts())

    train_naive_bayes(train_df, test_df)
    train_logistic_regression(train_df, test_df)
    train_textcnn(train_df, test_df)

    print("\nВСЕ МОДЕЛИ ОБУЧЕНЫ И СОХРАНЕНЫ.")


if __name__ == "__main__":
    main()
