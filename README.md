# Выполненный код первого задания

## Пример вывода программы:
```
Accuracy: 0.8483
Примеры прогнозов модели:

Текст: film adapted comic book plenty success , whether 're superheroes ( batman , superman , spawn ) , geared toward kid ( casper ) arthouse crowd ( ghost w...
Истинная метка: Положительный
Предсказанная метка: Положительный

Текст: every movie come along suspect studio , every indication stinker , everybody 's surprise ( perhaps even studio ) film becomes critical darling . mtv f...
Истинная метка: Положительный
Предсказанная метка: Положительный

Текст: 've got mail work alot better deserves . order make film success , cast two extremely popular attractive star , share screen two hour collect profit ....
Истинная метка: Положительный
Предсказанная метка: Негативный

Текст: `` jaw `` rare film grab attention show single image screen . movie open blackness , distant , alien-like underwater sound . come , first ominous bar ...
Истинная метка: Положительный
Предсказанная метка: Положительный

Текст: moviemaking lot like general manager nfl team post-salary cap era -- 've got know allocate resource . every dollar spent free-agent defensive tackle o...
Истинная метка: Положительный
Предсказанная метка: Положительный

Process finished with exit code 0
```

## Текст задания:

Вам предоставлен корпус текстовых аннотаций к фильмам, извлеченный из коллекции movie_reviews библиотеки NLTK.

Каждая аннотация классифицирована как "благоприятная" (обозначение: positive) или "негативная" (обозначение: negative).



Необходимо реализовать следующее:

Примените инструментарий NLTK для предварительной обработки текстового материала: Выполните сегментацию лексических единиц, используя метод word_tokenize. Исключите из текста так называемые "шумовые слова" (например, "the", "and", "is"), доступные в арсенале NLTK. Проведите нормализацию лексем до их базовой формы с помощью механизма WordNetLemmatizer. Сформируйте числовые характеристики для задачи классификации, опираясь на методику "мешка слов" (Bag of Words), которая подсчитывает частотность лексических единиц в текстах. 

Постройте и обучите модель логистической регрессии, реализованную в библиотеке sklearn, чтобы прогнозировать эмоциональный оттенок рецензии ("благоприятный" или "негативный"). Оцените эффективность модели на выделенной тестовой подвыборке, вычислив показатель точности (accuracy).  + Дополнительно выведите несколько примеров прогнозов модели для демонстрации её работы.

Технические требования:

- Используйте Python с библиотеками NLTK и scikit-learn. 
- Сохраните реализацию в файле с названием film_sentiment_evaluator.py. 
- Создайте публичный репозиторий на платформе GitHub, загрузите код и предоставьте ссылку на него. 
- В выводе программы должны присутствовать: Числовой показатель точности модели. 
- Не менее трёх случайно выбранных примеров, где указаны текст рецензии (в обработанном виде), истинный класс и предсказанный класс.