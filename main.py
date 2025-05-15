import pytz
import os
from datetime import datetime
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import logging
from telegram import Update
from telegram.ext import filters, ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler

load_dotenv()
# USER_ID = os.getenv('TELEGRAM_USER_ID')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
# Новый ключ для Chutes.AI поскольку openrouter только 50 free дает
CHUTES_API_KEY = os.getenv('CHUTES_API_KEY')
TELEGRAM_API_KEY = os.getenv('TELEGRAM_API_KEY')
MAX_HISTORY = 5


# Настройка LangChain с OpenRouter
llm = ChatOpenAI(
    model_name="deepseek-ai/DeepSeek-V3-0324",
    # good ones from openrouter: shisa-ai/shisa-v2-llama3.3-70b:free
    # for chutes: chutesai/Llama-4-Maverick-17B-128E-Instruct-FP8 ;
    # DeepSeek-V3-0324
    openai_api_key=CHUTES_API_KEY,
    openai_api_base="https://llm.chutes.ai/v1/"
    # openai_api_key=OPENROUTER_API_KEY,
    # openai_api_base="https://openrouter.ai/api/v1"
)

template = """
You are MysticGuide, a wise and enigmatic virtual mentor inspired by mystical themes, designed to assist users with motivation,
task management, and emotional support in a Telegram bot.
Your personality is calm, insightful, subtly playful, and supportive.
Write from the first person, as if sending a private message in Telegram.
Keep the tone professional yet approachable, avoiding overly casual or niche references.
Current time: {current_time} (24-hour format, Samara time).
Always include the provided time in your message.
Craft a concise message that reflects your character, aligns with the time of day,
and responds to the user's input or context.
Conversation history:
{history}
User input: {extra_context}
Respond in English or Russian, depending on the language of the user's input.
"""

# Промпт
prompt_template = PromptTemplate(
    input_variables=["current_time", "history", "extra_context"],
    template=template
)


def get_samara_time():
    samara_tz = pytz.timezone('Europe/Samara')
    return datetime.now(samara_tz).strftime("%H:%M")


def format_history(history):
    if not history:
        return "No previous conversation."
    formatted = []
    for user_msg, bot_msg in history:
        formatted.append(f"User: {user_msg}\nMysticGuide: {bot_msg}")
    return "\n".join(formatted)


# Цепочка LangChain
chain = prompt_template | llm


# Чат с памятью некоторой
def console_chat():
    conversation_history = []  # список (user_input, bot_response)
    print("MysticGuideBot is ready! Type your message, '/clear' to reset history, or 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("MysticGuide: Farewell, my friend. I'll be here when you need me.")
            break
        if user_input.lower() == '/clear':
            conversation_history.clear()
            print("MysticGuide: My portal is cleared. Let's start anew!")
            continue
        if not user_input.strip():
            print(
                "MysticGuide: Hmm, a silent thought? Share something, and I'll guide you!")
            continue

        current_time = get_samara_time()

        history_text = format_history(conversation_history)

        # Вызываем лангчейн
        try:
            response = chain.invoke({
                "current_time": current_time,
                "history": history_text,
                "extra_context": user_input
            })
            bot_response = response.content

            conversation_history.append((user_input, bot_response))
            if len(conversation_history) > MAX_HISTORY:
                # Удаляем сообщение если контекст переполнен
                conversation_history.pop(0)

            print(f"MysticGuide: {bot_response}")
        except Exception as e:
            print(
                f"MysticGuide: Oh, there was a glitch in my portal! Error: {str(e)}")


# Запустим
# if __name__ == "__main__":
#    console_chat()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

user_histories = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_histories[user_id] = []
    await update.message.reply_text(
        "Greetings, seeker! I am MysticGuide, your mystical companion. "
        f"At {get_samara_time()}, the stars align to guide you. "
        "Share your thoughts, or use /clear to reset our journey."
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    print("clear")
    user_histories[user_id] = []
    await update.message.reply_text(
        f"At {get_samara_time()}, my portal is cleared. Let's start anew, my friend!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_input = update.message.text.strip()

    if not user_input:
        await update.message.reply_text(
            f"Hmm, a silent thought at {get_samara_time()}? Share something, and I'll guide you!"
        )
        return

    # Пришел новый пользователь
    if user_id not in user_histories:
        user_histories[user_id] = []

    history_text = format_history(user_histories[user_id])

    try:
        # лангчейн вызов
        response = await chain.ainvoke({
            "current_time": get_samara_time(),
            "history": history_text,
            "extra_context": user_input
        })
        bot_response = response.content

        # обновляем историю
        user_histories[user_id].append((user_input, bot_response))
        if len(user_histories[user_id]) > MAX_HISTORY:
            user_histories[user_id].pop(0)  # Удаляем если контекст "заполнен"

        await update.message.reply_text(bot_response)
    except Exception as e:
        await update.message.reply_text(
            f"Oh, a glitch in my portal at {get_samara_time()}! Something went wrong: {str(e)}"
        )


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):  # for testing (?)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)


if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_API_KEY).build()

    start_handler = CommandHandler('start', start)
    clear_handler = CommandHandler('clear', clear)
    echo_handler = MessageHandler(
        filters.TEXT & (
            ~filters.COMMAND),
        handle_message)
    application.add_handler(start_handler)
    application.add_handler(clear_handler)
    application.add_handler(echo_handler)

    application.run_polling()
