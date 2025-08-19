import torch
import pandas as pd
import re
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import textstat  # для метрик читаемости

# ===================== Настройки =====================
OUTPUT_FILE = "filtered_wb_feedbacks.parquet"
SAIGA_MODEL = "IlyaGusev/saiga_yandexgpt_8b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_ROWS = 10000  # начать с меньшего количества для тестирования

# ===================== Rule-Based Фильтрация =====================
def rule_based_filtering(df):
    """
    Быстрая фильтрация по правилам - эффективно отсеивает мусор
    """
    keep_mask = []
    
    for _, row in df.iterrows():
        review = str(row['text'])
        answer = str(row['answer'])
        
        # Пропускаем пустые или почти пустые
        if len(review.strip()) < 5 or len(answer.strip()) < 3:
            keep_mask.append(False)
            continue
        
        # Критерии отсеивания
        should_reject = (
            len(answer) > 500 or  # Слишком длинные ответы
            len(answer) < 10 or  # Слишком короткие ответы
            re.search(r'http|www|\.ru|\.com|\.net|\.org', answer.lower()) or  # Ссылки
            re.search(r'[0-9]{10,}', answer) or  # Телефоны/номера
            re.search(r'[a-f0-9]{32}', answer) or  # Хэши
            any(phrase in answer.lower() for phrase in [
                'спасибо за отзыв', 'благодарим за отзыв', 'извините за',
                'приносим извинения', 'обратитесь в поддержку', 'напишите нам',
                'позвоните нам', 'контактный телефон'
            ]) or
            answer.lower().strip() in ['ok', 'хорошо', 'понятно', 'ясно', 'спасибо', 'благодарю'] or
            len(set(answer)) < 5  # Мало уникальных символов
        )
        
        keep_mask.append(not should_reject)
    
    return df[keep_mask]

# ===================== Heuristic Фильтрация =====================
def heuristic_filtering(df):
    """
    Фильтрация на основе статистических метрик и эвристик
    """
    keep_mask = []
    
    for _, row in df.iterrows():
        answer = str(row['answer'])
        
        # Пропускаем слишком короткие
        if len(answer) < 15:
            keep_mask.append(False)
            continue
        
        # Вычисляем метрики
        words = answer.split()
        word_count = len(words)
        unique_words = len(set(words))
        diversity_ratio = unique_words / word_count if word_count > 0 else 0
        
        # Простые метрики качества
        has_questions = any('?' in word for word in words)
        has_explanations = any(len(word) > 8 for word in words)  # Длинные слова часто содержат смысл
        sentence_count = answer.count('.') + answer.count('!') + answer.count('?')
        
        # Композитный скоринг
        score = (
            min(len(answer), 200) / 200 * 0.2 +  # Длина ответа
            min(diversity_ratio * 2, 1) * 0.3 +  # Разнообразие слов
            (1 if has_questions else 0) * 0.1 +  # Вопросы - признак диалога
            (1 if has_explanations else 0) * 0.2 +  # Объяснения
            min(sentence_count / 5, 1) * 0.2  # Количество предложений
        )
        
        keep_mask.append(score > 0.5)  # Пороговое значение
    
    return df[keep_mask]

# ===================== Ensemble Фильтрация =====================
def ensemble_filtering(df):
    """
    Комбинированная фильтрация - самый надежный метод
    """
    print("Начало многоступенчатой фильтрации...")
    
    # Первый проход: быстрая rule-based фильтрация
    df_filtered_1 = rule_based_filtering(df)
    print(f"После rule-based: {len(df_filtered_1)} строк")
    
    # Второй проход: статистическая фильтрация
    df_filtered_2 = heuristic_filtering(df_filtered_1)
    print(f"После heuristic: {len(df_filtered_2)} строк")
    
    return df_filtered_2

# ===================== Функция классификации Saiga (опционально) =====================
def classify_with_saiga(df, sample_size=1000):
    """
    Дополнительная фильтрация с помощью Saiga для небольшой выборки
    """
    if len(df) == 0:
        return df
    
    # Берем выборку для финальной проверки
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    
    print(f"Загружаем Saiga для финальной проверки {len(sample_df)} строк...")
    
    tokenizer = AutoTokenizer.from_pretrained(SAIGA_MODEL, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        SAIGA_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    keep_mask = []
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Saiga проверка"):
        review = str(row['text'])
        answer = str(row['answer'])
        
        prompt = f"""Определи, является ли следующий ответ качественным и содержательным для обучения AI-ассистента:

Отзыв: "{review[:100]}"
Ответ: "{answer[:200]}"

Ответ должен быть только 'true' или 'false':"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        is_good = 'true' in generated_text.lower()
        keep_mask.append(is_good)
    
    # Применяем результаты к выборке
    sample_df = sample_df[pd.Series(keep_mask, index=sample_df.index)]
    
    # Объединяем с остальными данными
    remaining_df = df[~df.index.isin(sample_df.index)]
    final_df = pd.concat([sample_df, remaining_df])
    
    return final_df

# ===================== Главная функция =====================
def main():
    print("Загружаем wb-feedbacks...")
    try:
        wb = load_dataset("nyuuzyou/wb-feedbacks", split="train")
        wb = wb.filter(lambda x: (x.get("text") or "") != "" and (x.get("answer") or "") != "")
        
        # Ограничиваем количество строк
        if len(wb) > MAX_ROWS:
            wb = wb.select(range(MAX_ROWS))
        
        wb_df = pd.DataFrame(wb)
        print(f"Обрабатываем {len(wb_df)} строк...")
        
        # Многоступенчатая фильтрация
        filtered_df = ensemble_filtering(wb_df)
        
        # Опционально: дополнительная проверка Saiga для небольшой выборки
        if len(filtered_df) > 1000:
            filtered_df = classify_with_saiga(filtered_df, sample_size=500)
        
        print(f"После всей фильтрации: {len(filtered_df)} строк")
        
        if len(filtered_df) > 0:
            # Подготовка для обучения
            filtered_df["formatted_text"] = "Отзыв: " + filtered_df["text"] + "\nОтвет: " + filtered_df["answer"]
            
            # Сохраняем полные данные и отформатированные
            filtered_df.to_parquet(OUTPUT_FILE, index=False)
            print(f"✅ Сохранено {len(filtered_df)} строк в {OUTPUT_FILE}")
            
            # Дополнительно сохраняем только текст для обучения
            filtered_df[["formatted_text"]].to_parquet("training_data.parquet", index=False)
            print("✅ Сохранены данные для обучения")
        else:
            print("❌ Нет подходящих данных для сохранения")
            
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()