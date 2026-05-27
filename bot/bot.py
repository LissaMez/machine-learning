import os
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from predict import predict_all, format_predictions

load_dotenv()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для распознавания эмоций в русском тексте.\n\n"
        "Отправь мне любое сообщение, а я покажу предсказания "
        "трёх моделей:\n"
        "1. Naive Bayes\n"
        "2. Logistic Regression\n"
        "3. TextCNN\n\n"
        "Я умею распознавать 5 эмоций:\n"
        "joy — радость\n"
        "sadness — грусть\n"
        "surprise — удивление\n"
        "fear — страх\n"
        "anger — злость"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_text = update.message.text

    if not user_text or not user_text.strip():
        await update.message.reply_text("Пожалуйста, отправь непустой текст.")
        return

    try:
        predictions = predict_all(user_text)
        answer = format_predictions(predictions)
        await update.message.reply_text(answer)

    except FileNotFoundError:
        await update.message.reply_text(
            "Модели ещё не обучены.\n\n"
            "Сначала запусти в терминале:\n"
            "python bot/train_models.py"
        )

    except Exception as error:
        await update.message.reply_text(
            "Произошла ошибка при обработке текста.\n" f"Техническая информация: {error}"
        )


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        raise ValueError(
            "Не найден TELEGRAM_BOT_TOKEN.\n"
            "Создай файл .env в корне проекта и добавь туда токен."
        )

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Бот запущен.")
    app.run_polling()


if __name__ == "__main__":
    main()
