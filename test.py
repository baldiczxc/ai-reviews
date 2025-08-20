import os
import gc
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

set_seed(42)

# =========================
# 1. Загружаем отфильтрованный датасет
# =========================
INPUT_DATA = "filtered_reviews.parquet"
if not os.path.exists(INPUT_DATA):
    raise FileNotFoundError(f"❌ Нет файла {INPUT_DATA}, сначала запусти filter_with_saiga_batch.py")

df = pd.read_parquet(INPUT_DATA)
dataset = Dataset.from_pandas(df[["text"]], preserve_index=False)

# =========================
# 2. Модель и токенайзер
# =========================
model_name = "ai-forever/rugpt3medium_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

# =========================
# 3. LoRA адаптер
# =========================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# =========================
# 4. Токенизация
# =========================
MAX_LEN = 192

def tokenize(batch):
    add_eos = [t if t.endswith(tokenizer.eos_token) else t + tokenizer.eos_token for t in batch["text"]]
    out = tokenizer(add_eos, truncation=True, padding="max_length", max_length=MAX_LEN)
    out["labels"] = out["input_ids"].copy()
    return out

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"], batch_size=1000)
split = tokenized.train_test_split(test_size=0.05, seed=42)
train_ds, eval_ds = split["train"], split["test"]

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# =========================
# 5. Обучение
# =========================
training_args = TrainingArguments(
    output_dir="./review_responder_rugpt3_lora_saiga",
    logging_dir="./logs",
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    fp16=True,
    gradient_checkpointing=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to=None,
    torch_compile=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=collator
)

print("Начинаем обучение...")
trainer.train()

print("Сохраняем модель...")
model.save_pretrained("./review_responder_rugpt3_lora_saiga", safe_serialization=True)
tokenizer.save_pretrained("./review_responder_rugpt3_lora_saiga")
print("✅ Готово!")
