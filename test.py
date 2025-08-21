import pandas as pd
from datasets import Dataset
from transformers import (
    GPT2TokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)

# ===================== Данные =====================
df = pd.read_parquet("filtered_wb_feedbacks.parquet")

# Формат: "Отзыв: ... Ответ: ..."
dataset = Dataset.from_pandas(df[['formatted_text']])
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# ===================== Токенизатор =====================
# Возьмем готовый токенизатор GPT-2 (можно сделать свой через HuggingFace Tokenizers)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 не имеет паддинга

# ===================== Конфигурация модели =====================
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_embd=256,   # уменьшаем размерность, чтобы обучать с нуля быстрее
    n_layer=6,    # кол-во слоёв
    n_head=8,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

model = GPT2LMHeadModel(config)

# ===================== Токенизация =====================
def tokenize(batch):
    return tokenizer(batch["formatted_text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["formatted_text"])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# ===================== Тренировка =====================
training_args = TrainingArguments(
    output_dir="./wb_gpt2_from_scratch",
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,
    warmup_steps=200,
    logging_steps=50,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

trainer.train()

# ===================== Сохранение =====================
trainer.save_model("./wb_gpt2_from_scratch")
tokenizer.save_pretrained("./wb_gpt2_from_scratch")
