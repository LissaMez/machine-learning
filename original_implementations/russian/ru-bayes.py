import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

russian_stopwords = stopwords.words('russian')

EMOTIONS_MAP = {
    '0': 'joy',
    '1': 'sadness',
    '2': 'surprise',
    '3': 'fear',
    '4': 'anger'
}


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'[^а-яА-ЯёЁ ]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def fix_label(val):
    val = str(val)

    val = val.replace('[', '').replace(']', '').strip()

    if val == '' or val == 'nan':
        return None

    first_digit = val.split()[0]

    return first_digit


def train_and_test():
    print("ЗАГРУЗКА ДАННЫХ")

    train_df = pd.read_csv(
        'ru-train.txt',
        header=None,
        names=['text', 'labels', 'source'],
        encoding='utf-8'
    )

    test_df = pd.read_csv(
        'ru-test.txt',
        header=None,
        names=['text', 'labels', 'source'],
        encoding='utf-8'
    )

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    train_df = train_df.dropna()
    test_df = test_df.dropna()

    train_df['target_digit'] = train_df['labels'].apply(fix_label)
    test_df['target_digit'] = test_df['labels'].apply(fix_label)

    train_df = train_df.dropna(subset=['target_digit'])
    test_df = test_df.dropna(subset=['target_digit'])

    train_df['target_name'] = train_df['target_digit'].map(EMOTIONS_MAP)
    test_df['target_name'] = test_df['target_digit'].map(EMOTIONS_MAP)

    print("TF-IDF ВЕКТОРИЗАЦИЯ")

    tfidf = TfidfVectorizer(
        stop_words=russian_stopwords,
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train = tfidf.fit_transform(
        train_df['text'].apply(clean_text)
    )

    X_test = tfidf.transform(
        test_df['text'].apply(clean_text)
    )

    y_train = train_df['target_name']
    y_test = test_df['target_name']

    print(f"Количество признаков: {X_train.shape[1]}")

    print("ОБУЧЕНИЕ MULTINOMIAL NAIVE BAYES")

    nb_model = MultinomialNB(alpha=0.05)

    start_time = time.time()

    nb_model.fit(X_train, y_train)

    end_time = time.time()

    training_time = end_time - start_time

    print(f"Training time: {training_time:.2f} seconds")

    y_pred = nb_model.predict(X_test)

    y_probs = nb_model.predict_proba(X_test)

    max_probs = y_probs.max(axis=1)

    avg_confidence = max_probs.mean()

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average='macro'
    )

    print("РЕЗУЛЬТАТЫ МОДЕЛИ")

    print(f"\nAccuracy: {accuracy:.4f}")

    print(f"Macro F1-score: {macro_f1:.4f}")

    print(f"Average confidence: {avg_confidence:.2%}")

    print("\nClassification Report:\n")

    print(classification_report(y_test, y_pred))

    print("\nCONFUSION MATRIX:\n")

    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    labels = sorted(y_test.unique())

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='viridis',
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.title('Confusion Matrix - Naive Bayes')

    plt.show()

    y_probs = nb_model.predict_proba(X_test)

    max_probs = y_probs.max(axis=1)

    plt.figure(figsize=(8, 5))

    plt.hist(max_probs, bins=10)

    plt.title('Confidence Distribution (Russian Dataset)')

    plt.xlabel('Confidence')

    plt.ylabel('Number of Predictions')

    plt.tight_layout()

    plt.savefig('confidence_distribution_russian.png')

    plt.close()

    feature_names = tfidf.get_feature_names_out()

    classes = nb_model.classes_

    for i, emotion in enumerate(classes):
        top10 = nb_model.feature_log_prob_[i].argsort()[-10:]

        words = [feature_names[j] for j in top10]

        scores = nb_model.feature_log_prob_[i][top10]

        plt.figure(figsize=(8, 5))

        plt.barh(words, scores)

        plt.title(f'Top Words for {emotion} (Russian)')

        plt.xlabel('Importance')

        plt.tight_layout()

        plt.savefig(f'top_words_{emotion}_russian.png')

        plt.close()

    results_df = pd.DataFrame({
        'Text': test_df['text'],
        'Real_Emotion': y_test,
        'Predicted_Emotion': y_pred
    })

    results_df.to_csv(
        'ru_nb_test_results.csv',
        index=False,
        sep=';',
        encoding='utf-8'
    )

    print("\nВсе результаты сохранены:")
    print("ru_nb_test_results.csv")

    errors_df = results_df[
        results_df['Real_Emotion']
        != results_df['Predicted_Emotion']
        ]

    errors_df.to_csv(
        'ru_nb_errors.csv',
        index=False,
        sep=';',
        encoding='utf-8'
    )

    print("\nОшибки модели сохранены:")
    print("ru_nb_errors.csv")


if __name__ == "__main__":
    train_and_test()

    print("\nПрограмма успешно завершена.")
