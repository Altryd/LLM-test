from typing import List
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, movie_reviews
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


def calculate_metrics(predictions: np.ndarray,
                      Y_test: np.ndarray) -> float:
    """
    Вычисляет и возвращает метрику accuracy
    :param predictions: прогнозы, полученные от модели
    :param Y_test: истинные значения прогнозов
    :return:
    """
    return (predictions == Y_test).mean()


class Tokenizer:
    def __init__(
            self
    ) -> None:
        self.lemmatizer = WordNetLemmatizer()

    @classmethod
    def eliminate_stopwords(
            cls, tokenized_list: List["str"], stopwords=stopwords.words('english')):
        output = []
        for token in tokenized_list:
            if token in stopwords:
                continue
            output.append(token)
        return output

    def tokenize_text(self, text):
        tokens = nltk.word_tokenize(text.lower())
        tokens = Tokenizer.eliminate_stopwords(tokens)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)


if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('movie_reviews')
    nltk.download('wordnet')

    train_ratio = 0.7
    pos_len, neg_len = len(
        movie_reviews.fileids('pos')), len(
        movie_reviews.fileids('neg'))
    train_dataset_ids = movie_reviews.fileids('pos')[0:int(
        train_ratio * pos_len)] + movie_reviews.fileids('neg')[0:int(train_ratio * neg_len)]
    test_dataset_ids = movie_reviews.fileids('pos')[int(
        train_ratio * pos_len):] + movie_reviews.fileids('neg')[int(train_ratio * neg_len):]
    train_dataset = []  # first item - id, second item will be label
    test_dataset = []
    for _id in train_dataset_ids:
        if "pos" in _id:
            train_dataset.append([_id, 1])
        elif "neg" in _id:
            train_dataset.append([_id, 0])

    for _id in test_dataset_ids:
        if "pos" in _id:
            test_dataset.append([_id, 1])
        elif "neg" in _id:
            test_dataset.append([_id, 0])

    tokenizer = Tokenizer()
    X_train_tokenized = [
        tokenizer.tokenize_text(
            movie_reviews.raw(
                item[0])) for item in train_dataset]
    Y_train = np.array([item[1] for item in train_dataset])
    X_test_tokenized = [
        tokenizer.tokenize_text(
            movie_reviews.raw(
                item[0])) for item in test_dataset]
    Y_test = np.array([item[1] for item in test_dataset])

    vectorizer = CountVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(X_train_tokenized).toarray()
    X_test = vectorizer.transform(X_test_tokenized).toarray()

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, Y_train)

    predictions = clf.predict(X_test)
    accuracy = calculate_metrics(predictions, Y_test)
    print(f"Accuracy: {accuracy:.4f}")

    print("Примеры прогнозов модели:")
    for i in range(5):  # Первые 5 примеров
        text_snippet = X_test_tokenized[i][:150] + "..."
        true_label = "Положительный" if Y_test[i] == 1 else "Негативный"
        pred_label = "Положительный" if predictions[i] == 1 else "Негативный"
        print(f"\nТекст: {text_snippet}")
        print(f"Истинная метка: {true_label}")
        print(f"Предсказанная метка: {pred_label}")
