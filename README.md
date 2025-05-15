### На самом деле не совсем понял что имеется в виду под "локально загрузить модельку", поэтому просто через LangChain до бесплатного провайдера иду
## Установка
Тестировалось на python 3.11.11 (conda env)
1. Заменить `.env.example` на `.env`:
   ```bash
   cp .env.example .env

2. Заполнить в .env (пока что используется CHUTES, но в коде можно переправить чтобы использовался OPENROUTER)

OPENROUTER_API_KEY=

CHUTES_API_KEY=

TELEGRAM_API_KEY=

3. ```bash
   pip install -r requirements.txt
   python main.py