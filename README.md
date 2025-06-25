## Установка
Тестировалось на python 3.11.11 (conda env)

1. ```bash
   pip install -r requirements.txt
   python main.py

## Пример вывода:
Map: 100%|██████████| 80/80 [00:00<00:00, 678.72 examples/s]
Map: 100%|██████████| 20/20 [00:00<00:00, 624.96 examples/s]
 20%|██        | 10/50 [00:17<01:06,  1.67s/it]{'loss': 0.6957, 'grad_norm': 1.9785504341125488, 'learning_rate': 9e-06, 'epoch': 1.0}

  0%|          | 0/3 [00:00<?, ?it/s]
 67%|██████▋   | 2/3 [00:00<00:00,  4.26it/s]
                                               
 20%|██        | 10/50 [00:18<01:06,  1.67s/it]
100%|██████████| 3/3 [00:00<00:00,  4.15it/s]
                                             {'eval_loss': 0.6807359457015991, 'eval_accuracy': 0.7, 'eval_f1': 0.7692307692307693, 'eval_precision': 0.625, 'eval_recall': 1.0, 'eval_runtime': 1.1934, 'eval_samples_per_second': 16.759, 'eval_steps_per_second': 2.514, 'epoch': 1.0}
 40%|████      | 20/50 [00:36<00:49,  1.66s/it]{'loss': 0.6744, 'grad_norm': 2.670250654220581, 'learning_rate': 1.9e-05, 'epoch': 2.0}

  0%|          | 0/3 [00:00<?, ?it/s]
 67%|██████▋   | 2/3 [00:00<00:00,  4.16it/s]
                                               
 40%|████      | 20/50 [00:37<00:49,  1.66s/it]
100%|██████████| 3/3 [00:00<00:00,  4.15it/s]
                                             {'eval_loss': 0.6597973108291626, 'eval_accuracy': 0.75, 'eval_f1': 0.7058823529411765, 'eval_precision': 0.8571428571428571, 'eval_recall': 0.6, 'eval_runtime': 1.2001, 'eval_samples_per_second': 16.665, 'eval_steps_per_second': 2.5, 'epoch': 2.0}
 60%|██████    | 30/50 [00:55<00:34,  1.73s/it]{'loss': 0.6111, 'grad_norm': 2.6435365676879883, 'learning_rate': 2.9e-05, 'epoch': 3.0}

  0%|          | 0/3 [00:00<?, ?it/s]
 67%|██████▋   | 2/3 [00:00<00:00,  4.09it/s]
{'eval_loss': 0.5618258118629456, 'eval_accuracy': 0.85, 'eval_f1': 0.8571428571428571, 'eval_precision': 0.8181818181818182, 'eval_recall': 0.9, 'eval_runtime': 1.1942, 'eval_samples_per_second': 16.748, 'eval_steps_per_second': 2.512, 'epoch': 3.0}
                                               
 60%|██████    | 30/50 [00:56<00:34,  1.73s/it]
100%|██████████| 3/3 [00:00<00:00,  4.12it/s]
 80%|████████  | 40/50 [01:14<00:18,  1.83s/it]{'loss': 0.4025, 'grad_norm': 2.542452096939087, 'learning_rate': 3.9000000000000006e-05, 'epoch': 4.0}

  0%|          | 0/3 [00:00<?, ?it/s]
 67%|██████▋   | 2/3 [00:00<00:00,  4.18it/s]
                                               
 80%|████████  | 40/50 [01:15<00:18,  1.83s/it]
100%|██████████| 3/3 [00:00<00:00,  4.14it/s]
                                             {'eval_loss': 0.31111225485801697, 'eval_accuracy': 0.95, 'eval_f1': 0.9473684210526315, 'eval_precision': 1.0, 'eval_recall': 0.9, 'eval_runtime': 1.2181, 'eval_samples_per_second': 16.419, 'eval_steps_per_second': 2.463, 'epoch': 4.0}
100%|██████████| 50/50 [01:33<00:00,  1.72s/it]{'loss': 0.1309, 'grad_norm': 0.9674394130706787, 'learning_rate': 4.9e-05, 'epoch': 5.0}

  0%|          | 0/3 [00:00<?, ?it/s]
 67%|██████▋   | 2/3 [00:00<00:00,  4.24it/s]
                                               
100%|██████████| 50/50 [01:35<00:00,  1.72s/it]
100%|██████████| 3/3 [00:00<00:00,  4.17it/s]
                                             {'eval_loss': 0.16086706519126892, 'eval_accuracy': 0.9, 'eval_f1': 0.8888888888888888, 'eval_precision': 1.0, 'eval_recall': 0.8, 'eval_runtime': 1.1988, 'eval_samples_per_second': 16.684, 'eval_steps_per_second': 2.503, 'epoch': 5.0}
{'train_runtime': 96.0051, 'train_samples_per_second': 4.166, 'train_steps_per_second': 0.521, 'train_loss': 0.5028988480567932, 'epoch': 5.0}
100%|██████████| 50/50 [01:36<00:00,  1.92s/it]

Результаты классификации новых отзывов:
Отзыв: The food was absolutely delicious and the service was top-notch!
Sentiment: POSITIVE

Отзыв: Worst experience ever, the staff was rude and the food was cold.
Sentiment: NEGATIVE

Отзыв: Amazing ambiance, but food could be better
Sentiment: POSITIVE

Отзыв: I loved the dessert, will definitely come back for more!
Sentiment: POSITIVE

Отзыв: Terrible, the place was dirty and noisy.
Sentiment: NEGATIVE

Отзыв: Great atmosphere and friendly staff, highly recommend!
Sentiment: POSITIVE

Отзыв: Disappointing, waited over an hour for food.
Sentiment: NEGATIVE

Отзыв: The pasta was divine, best I've ever had!
Sentiment: POSITIVE

Отзыв: Overpriced for such bland food.
Sentiment: NEGATIVE

Отзыв: Wonderful experience, everything was perfect!
Sentiment: POSITIVE