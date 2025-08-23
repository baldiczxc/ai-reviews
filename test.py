import os
import pandas as pd
from datasets import Dataset
import sentencepiece as spm
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Trainer,
)
import torch
from pathlib import Path
from typing import List

print(torch.cuda.is_available())       # должно быть True
print(torch.cuda.device_count())       # количество GPU
print(torch.cuda.get_device_name(0))   # имя первого GPU
def main():
    # ===================== Пути и базовые настройки =====================
    DATA_FILE = "filtered_wb_feedbacks.parquet"  # твой файл
    WORKDIR = Path("./t5_scratch_run")           # рабочая папка эксперимента
    TOKENIZER_DIR = WORKDIR / "tokenizer"
    SPM_MODEL = TOKENIZER_DIR / "spiece.model"
    MODEL_OUT = WORKDIR / "model"
    CORPUS_TXT = WORKDIR / "corpus.txt"

    MAX_INPUT_LEN = 256     # длина входа (отзыв)
    MAX_TARGET_LEN = 128    # длина ответа (реплика магазина)

    # ====== размерность "маленькой" T5 (под 1 GPU). Увеличивай по мере возможностей ======
    VOCAB_SIZE = 32000      # словарь токенайзера (часто 16k–50k)
    D_MODEL = 384           # 512/768+ лучше, но дороже
    D_FF = 1536
    NUM_LAYERS = 6          # 8–12 для серьёзной модели, но тут 6, чтобы взлетело
    NUM_HEADS = 6           # D_MODEL должен делиться на NUM_HEADS
    DROPOUT = 0.1

    LR = 3e-4
    EPOCHS = 5
    BATCH_TRAIN = 8
    BATCH_EVAL = 8
    GRAD_ACCUM = 4          # эффективный batch = 8*4=32
    WARMUP_RATIO = 0.03
    WEIGHT_DECAY = 0.01

    # ===================== 0) Подготовка окружения =====================
    WORKDIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    # ===================== 1) Загрузка и подготовка корпуса =====================
    print("Загружаем датасет…")
    df = pd.read_parquet(DATA_FILE)

    # Берём очищенные поля; отфильтруем NaN и совсем короткие строки
    df = df[["text_cleaned", "answer_cleaned"]].dropna()
    df = df[(df["text_cleaned"].str.len() > 10) & (df["answer_cleaned"].str.len() > 10)]

    # Собираем общий корпус для обучения токенайзера:
    # в одном файле попеременно строки отзывов и ответов
    print("Готовим текстовый корпус для токенайзера…")
    with open(CORPUS_TXT, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(str(row["text_cleaned"]).strip() + "\n")
            f.write(str(row["answer_cleaned"]).strip() + "\n")

    # ===================== 2) Обучение SentencePiece (Unigram) =====================
    # Важно задать спец-токены и их id, чтобы они совпадали с ожиданиями T5
    # pad_id=0, eos_id=1, unk_id=2, bos_id=-1 (T5 не использует BOS)
    print("Обучаем SentencePiece токенайзер…")
    spm.SentencePieceTrainer.Train(
        input=str(CORPUS_TXT),
        model_prefix=str(SPM_MODEL).replace(".model", ""),
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        character_coverage=0.9995,   # для рус/eng/символов
        num_threads=os.cpu_count() or 4,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
        normalization_rule_name="identity", 
        user_defined_symbols=[],     # можно добавить спец-теги, если нужны
    )

    assert SPM_MODEL.exists(), "SentencePiece модель не создалась"

    # ===================== 3) Инициализация T5Tokenizer на своём spiece.model =====================
    print("Инициализируем T5Tokenizer…")
    tokenizer = T5Tokenizer(vocab_file=str(SPM_MODEL))
    # Важно: явно зафиксируем спец-токены
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    # Сохраняем токенайзер как папку
    tokenizer.save_pretrained(str(TOKENIZER_DIR))

    # ===================== 4) Создание HF Dataset и разбиение =====================
    print("Готовим HF Dataset…")
    hf_ds = Dataset.from_pandas(df.reset_index(drop=True))
    hf_ds = hf_ds.train_test_split(test_size=0.1, seed=42)

    # ===================== 5) Токенизация выборок =====================
    def preprocess(batch):
        # Можно добавить промпт, но для обучения с нуля чаще простой формат:
        # ВХОД: отзыв; ЦЕЛЬ: ответ
        inputs: List[str] = ["Отзыв: " + x for x in batch["text_cleaned"]]
        targets: List[str] = batch["answer_cleaned"]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding=False,
            )

        model_inputs["labels"] = labels["input_ids"]

        # заменим пад-ид в labels на -100, чтобы не учитывались в лоссе
        model_inputs["labels"] = [
            [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
            for seq in model_inputs["labels"]
        ]
        return model_inputs

    print("Токенизируем…")
    tokenized = hf_ds.map(
        preprocess,
        batched=True,
        remove_columns=hf_ds["train"].column_names,
        desc="Tokenizing",
    )

    # ===================== 6) Конфиг новой T5 "с нуля" =====================
    print("Создаём T5Config…")
    config = T5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT,
        layer_norm_epsilon=1e-6,
        feed_forward_proj="relu",    # можно "gated-gelu" как в T5 v1.1, если хочешь
        # Спец-токены
        pad_token_id=tokenizer.pad_token_id,   # 0
        eos_token_id=tokenizer.eos_token_id,   # 1
        # T5 обычно стартует декодер с pad_token_id
        decoder_start_token_id=tokenizer.pad_token_id,
    )

    # ===================== 7) Проверка GPU и модель с нуля =====================
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Используем GPU: {torch.cuda.get_device_name(0)}")

    print("Инициализируем T5ForConditionalGeneration (с нуля)…")
    model = T5ForConditionalGeneration(config).to(device)


    # Полезно: bf16/amp, если доступно
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # ===================== 8) Тренировочные параметры =====================
    print("Готовим TrainingArguments…")
    args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_OUT),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        logging_dir=str(WORKDIR / "logs"),
        logging_steps=100,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=4,
        report_to="none",  # включи 'tensorboard' при желании
        predict_with_generate=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ===================== 9) Тренировка =====================
    print("Старт обучения…")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # ===================== 10) Сохранение =====================
    print("Сохраняем модель и токенайзер…")
    trainer.save_model(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))

    print("Готово ✅")
    print(f"Модель: {MODEL_OUT}")
    print(f"Токенайзер: {TOKENIZER_DIR}")

if __name__ == "__main__":
    main()