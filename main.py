import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from peft import LoraConfig, get_peft_model


def tokenize_function(examples):
    inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs


def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', names=['question', 'answer'])
    data['text'] = data.apply(lambda row: f"Вопрос: {row['question']} Ответ: {row['answer']}", axis=1)
    return Dataset.from_pandas(data)


if __name__ == '__main__':
    file_path = "support_responses.txt"
    dataset = load_data(file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Using device: {device}")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['question', 'answer', 'text'])
    tokenized_dataset.set_format("torch")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./gpt2-lora-support",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=100,
        learning_rate=5e-5,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    # save the model and tokenizer explicitly
    model_output_dir = './gpt2-support'

    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
