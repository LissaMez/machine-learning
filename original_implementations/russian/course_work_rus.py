import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import re
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import time

plt.rcParams["font.family"] = "DejaVu Sans"

label_map = {0: "joy", 1: "sadness", 2: "surprise", 3: "fear", 4: "anger"}


def read_emotion_file(path, has_header=False):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        if has_header:
            next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            text = row[0].strip()
            labels_raw = row[1].strip()
            if labels_raw == "[]":
                continue
            labels = re.findall(r"\d+", labels_raw)
            if len(labels) != 1:
                continue
            label_code = int(labels[0])
            rows.append(
                {"text": text, "label": label_map[label_code], "label_code": label_code}
            )
    return pd.DataFrame(rows)


FOLDER_PATH = r"C:\Users\Пользователь\Desktop\coursework_emotion"

print("ЗАГРУЗКА ДАННЫХ")
train_path = os.path.join(FOLDER_PATH, "ru-train.txt")
test_path = os.path.join(FOLDER_PATH, "ru-test.txt")

print(f"Тренировочный файл: {train_path}")
print(f"Тестовый файл: {test_path}")

if not os.path.exists(train_path):
    print(f"ОШИБКА: файл не найден - {train_path}")
    exit()
if not os.path.exists(test_path):
    print(f"ОШИБКА: файл не найден - {test_path}")
    exit()

train_df = read_emotion_file(train_path, has_header=True)
test_df = read_emotion_file(test_path, has_header=False)

print(f"Train: {len(train_df)} samples")
print(f"Test:  {len(test_df)} samples")

print("ВЕКТОРИЗАЦИЯ ТЕКСТА")

vectorizer = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.9
)

X_train = vectorizer.fit_transform(train_df["text"].values)
X_test = vectorizer.transform(test_df["text"].values)
y_train = train_df["label"].values
y_test = test_df["label"].values

print(f"Размер матрицы признаков: {X_train.shape}")


print("ОБУЧЕНИЕ МОДЕЛИ")

clf = LogisticRegression(
    C=1.0, max_iter=1000, solver="lbfgs", random_state=42, verbose=1
)

start = time.time()
clf.fit(X_train, y_train)
print(f"Обучение завершено за {time.time() - start:.1f} сек")

print("ПРЕДСКАЗАНИЕ")

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"F1 macro:  {f1_macro:.4f}")
print(f"\n{classification_report(y_test, y_pred)}")


print("МАТРИЦА ОШИБОК (текстовый вид)")


cm = confusion_matrix(y_test, y_pred)
class_names = clf.classes_

print("\n" + " " * 12, end="")
for name in class_names:
    print(f"{name:>10}", end="")
print()


for i, name in enumerate(class_names):
    print(f"{name:>12}", end="")
    for j in range(len(class_names)):
        print(f"{cm[i][j]:>10}", end="")
    print()

print("\nРасшифровка: строки - реальные эмоции, столбцы - предсказанные")


with open(
    os.path.join(FOLDER_PATH, "confusion_matrix_russian.txt"), "w", encoding="utf-8"
) as f:
    f.write("Матрица ошибок - Русский датасет\n")
    f.write("=" * 50 + "\n")
    f.write(" " * 12)
    for name in class_names:
        f.write(f"{name:>10}")
    f.write("\n")
    for i, name in enumerate(class_names):
        f.write(f"{name:>12}")
        for j in range(len(class_names)):
            f.write(f"{cm[i][j]:>10}")
        f.write("\n")
    f.write("\nРасшифровка: строки - реальные эмоции, столбцы - предсказанные\n")

print(
    f"\nТекстовая матрица сохранена: {os.path.join(FOLDER_PATH, 'confusion_matrix_russian.txt')}"
)


print("ВИЗУАЛИЗАЦИЯ МАТРИЦЫ ОШИБОК (график)")


russian_names = {
    "anger": "гнев",
    "fear": "страх",
    "joy": "радость",
    "sadness": "печаль",
    "surprise": "удивление",
}
class_names_ru = [russian_names.get(name, name) for name in class_names]

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names_ru,
    yticklabels=class_names_ru,
)
plt.xlabel("Предсказанные эмоции", fontsize=12)
plt.ylabel("Реальные эмоции", fontsize=12)
plt.title("Матрица ошибок - Русский датасет", fontsize=14)
plt.tight_layout()


image_path = os.path.join(FOLDER_PATH, "confusion_matrix_russian.png")
plt.savefig(image_path, dpi=150)
print(f"График матрицы ошибок сохранён: {image_path}")


plt.show()

print("\nГОТОВО!")
