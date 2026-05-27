import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score
)
import time
import matplotlib.pyplot as plt
import seaborn as sns

try:
    train_df = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])
    test_df = pd.read_csv('test.txt', sep=';', header=None, names=['text', 'emotion'])
    print("Данные успешно загружены")
    print(f"Строк в обучении: {len(train_df)}, строк в тесте: {len(test_df)}")
except Exception as e:
    print(f"Ошибка при загрузке файлов: {e}")
    exit()

train_df = train_df.dropna()
test_df = test_df.dropna()

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

X_train = tfidf.fit_transform(train_df['text'])

X_test = tfidf.transform(test_df['text'])

y_train = train_df['emotion']
y_test = test_df['emotion']

nb_model = MultinomialNB()

start_time = time.time()

nb_model.fit(X_train, y_train)

end_time = time.time()

print(f"\nTraining time: {end_time - start_time:.2f} seconds")

y_pred = nb_model.predict(X_test)

print("РЕЗУЛЬТАТЫ NAIVE BAYES")

print(f"Общая точность (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nДетальный отчет по метрикам:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

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
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"Macro F1-score: {macro_f1:.4f}")
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

y_probs = nb_model.predict_proba(X_test)

max_probs = y_probs.max(axis=1)

plt.figure(figsize=(8, 5))

plt.hist(max_probs, bins=10)

plt.title('Confidence Distribution (English Dataset)')

plt.xlabel('Confidence')

plt.ylabel('Number of Predictions')

plt.tight_layout()

plt.savefig('confidence_distribution_english.png')

plt.close()

feature_names = tfidf.get_feature_names_out()

classes = nb_model.classes_

for i, emotion in enumerate(classes):
    top10 = nb_model.feature_log_prob_[i].argsort()[-10:]

    words = [feature_names[j] for j in top10]

    scores = nb_model.feature_log_prob_[i][top10]

    plt.figure(figsize=(8, 5))

    plt.barh(words, scores)

    plt.title(f'Top Words for {emotion} (English)')

    plt.xlabel('Importance')

    plt.tight_layout()

    plt.savefig(f'top_words_{emotion}_english.png')

    plt.close()

results_df = pd.DataFrame({
    'Text': test_df['text'],
    'Real_Emotion': y_test,
    'Predicted_Emotion': y_pred
})

results_df.to_csv('nb_test_results.csv', index=False, sep=';', encoding='utf-8')
print("\nРезультаты теста сохранены в файл 'nb_test_results.csv'")

errors_df = results_df[
    results_df['Real_Emotion'] != results_df['Predicted_Emotion']
    ]

errors_df.to_csv(
    'nb_errors.csv',
    index=False,
    sep=';',
    encoding='utf-8'
)

print("Ошибки модели сохранены в файл 'nb_errors.csv'")
