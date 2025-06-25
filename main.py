import numpy as np
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=256
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def select_balanced(dataset, num_per_class):
    pos_indices = [i for i, label in enumerate(dataset['label']) if label == 1][:num_per_class]
    neg_indices = [i for i, label in enumerate(dataset['label']) if label == 0][:num_per_class]
    indices = pos_indices + neg_indices
    return dataset.select(indices)


if __name__ == '__main__':
    dataset = load_dataset("fancyzhx/yelp_polarity")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    train_dataset = select_balanced(dataset['train'], 40)
    test_dataset = select_balanced(dataset['test'], 10)

    dataset_small = {
        'train': train_dataset.shuffle(seed=5252),
        'test': test_dataset.shuffle(seed=5252)
    }

    tokenized_dataset = {
        'train': dataset_small['train'].map(tokenize_function, batched=True),
        'test': dataset_small['test'].map(tokenize_function, batched=True)
    }

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics
    )

    trainer.train()

    new_reviews = [
        "The food was absolutely delicious and the service was top-notch!",
        "Worst experience ever, the staff was rude and the food was cold.",
        "Amazing ambiance, but food could be better",
        "I loved the dessert, will definitely come back for more!",
        "Terrible, the place was dirty and noisy.",
        "Great atmosphere and friendly staff, highly recommend!",
        "Disappointing, waited over an hour for food.",
        "The pasta was divine, best I've ever had!",
        "Overpriced for such bland food.",
        "Wonderful experience, everything was perfect!"
    ]

    tokenized_reviews = tokenizer(
        new_reviews,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized_reviews)
        predictions = torch.argmax(outputs.logits, dim=-1).numpy()

    print("\nРезультаты классификации новых отзывов:")
    for review, pred in zip(new_reviews, predictions):
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"Отзыв: {review}\nSentiment: {sentiment}\n")

    model.save_pretrained('./sentiment_model')
    tokenizer.save_pretrained('./sentiment_model')
