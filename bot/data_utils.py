import csv
import re
import pandas as pd

LABEL_MAP = {0: "joy", 1: "sadness", 2: "surprise", 3: "fear", 4: "anger"}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^а-яё ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_russian_emotion_file(path, has_header=False):

    rows = []

    with open(path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)

        if has_header:
            next(reader, None)

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

            if label_code not in LABEL_MAP:
                continue

            rows.append(
                {
                    "text": clean_text(text),
                    "label": LABEL_MAP[label_code],
                    "label_code": label_code,
                }
            )

    return pd.DataFrame(rows)
