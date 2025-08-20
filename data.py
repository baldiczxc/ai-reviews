import pandas as pd
import re
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import os

# ===================== Настройки =====================
OUTPUT_FILE = "filtered_wb_feedbacks.parquet"
CHUNK_SIZE = 150000  # Обрабатываем по 50к строк за раз
TEMP_DIR = "temp_chunks"  # Папка для временных файлов

# Создаем временную папку
os.makedirs(TEMP_DIR, exist_ok=True)

# ===================== Функция очистки от персональных данных =====================
def remove_personal_data(text):
    """Удаляет имена, фамилии и другие персональные данные"""
    if not isinstance(text, str):
        return text
    
    # Удаляем имена в формате "Имя, добрый день"
    text = re.sub(r'^[А-ЯЁ][а-яё]+[,!.]?\s+(добрый|привет|здравствуйте|извините)', '', text, flags=re.IGNORECASE)
    
    # Удаляем обращения с именами
    text = re.sub(r'(уважаемы[ей]|дорог[ойая])\s+[А-ЯЁ][а-яё]+', '', text, flags=re.IGNORECASE)
    
    # Удаляем email адреса
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    # Удаляем номера телефонов
    text = re.sub(r'(\+7|8)[\s\-\(\)]*\d{3}[\s\-\(\)]*\d{3}[\s\-\(\)]*\d{2}[\s\-\(\)]*\d{2}', '[PHONE]', text)
    
    return text.strip()

# ===================== Умная фильтрация для чанка =====================
def process_chunk(chunk_data):
    """Обрабатывает один чанк данных"""
    
    # Расширенный список стоп-слов для рекламы
    ad_phrases = [
        'ждем вас снова', 'ждём вас снова', 'ждем за покупками', 'ждём за покупками',
        'приятных покупок', 'надеемся увидеть вас снова', 'с надеждой видеть', 
        'будем рады видеть', 'ждем в нашем магазине', 'ждём в нашем магазине',
        'покупай', 'заказывай', 'приобретай', 'скидк', 'выгодн', 'акци', 
        'специальное предложение', 'лучшее предложение', 'успейте заказать', 
        'успейте купить', 'подарок каждому', 'каталог', 'ассортимент',
        'рекомендуем попробовать', 'советуем попробовать', 'обратите внимание на', 
        'посмотрите также', 'наши сотрудники', 'наши подруги', 'мы уверены что',
        'уверены что вы сможете подобрать'
    ]

    results = []
    cleaned_texts = []
    cleaned_answers = []
    filtered_data = []

    for item in chunk_data:
        original_text = str(item.get('text', ''))
        original_answer = str(item.get('answer', ''))

        # Очищаем от персональных данных
        cleaned_text = remove_personal_data(original_text)
        cleaned_answer = remove_personal_data(original_answer)

        # Пропускаем пустые
        if len(cleaned_text) < 10 or len(cleaned_answer) < 10:
            continue

        answer_lower = cleaned_answer.lower()

        # 1. ЖЕСТКИЕ КРИТЕРИИ ОТСЕИВАНИЯ
        is_advertisement = (
            re.search(r'арт\.\s*\d+|артикул\s*\d+', answer_lower) or
            re.search(r'рекомендуем.*\d+|предлагаем.*\d+', answer_lower) or
            re.search(r'[#№]\d+', answer_lower) or
            any(phrase in answer_lower for phrase in ad_phrases)
        )
        
        is_template = (
            len(cleaned_answer) < 25 or
            (answer_lower.startswith(('спасибо', 'благодар', 'извин')) and len(cleaned_answer) < 50) or
            any(phrase in answer_lower for phrase in [
                'спасибо за отзыв', 'благодарим за отзыв', 'обратитесь в поддержку',
                'напишите нам', 'позвоните нам', 'контактный телефон',
                'ваш отзыв очень важен', 'будем рады помочь', 'приносим извинения'
            ])
        )

        has_junk = (
            re.search(r'http|www|\.ru|\.com', cleaned_answer) or
            re.search(r'[0-9]{10,}', cleaned_answer) or
            '[EMAIL]' in cleaned_answer or
            '[PHONE]' in cleaned_answer
        )

        # 2. КРИТЕРИИ КАЧЕСТВЕННОГО ОТВЕТА
        text_words = set(word for word in cleaned_text.lower().split() if len(word) > 4)
        answer_words = set(cleaned_answer.lower().split())
        common_words = text_words & answer_words
        has_relevance = len(common_words) > 0

        has_quality = (
            len(cleaned_answer.split()) > 8 and
            any(char in cleaned_answer for char in '.!?') and
            any(word in answer_lower for word in [
                'качеств', 'доставк', 'размер', 'цвет', 'материал',
                'гаранти', 'обмен', 'возврат', 'рекоменд', 'совету',
                'проблем', 'решен', 'изменен', 'улучшен', 'исправлен'
            ]) and
            not any(ad_phrase in answer_lower for ad_phrase in ['купит', 'заказ', 'покупк', 'магазин', 'ассортимент'])
        )

        # 3. ФИНАЛЬНОЕ РЕШЕНИЕ
        should_keep = (
            has_relevance and
            has_quality and
            not is_template and
            not has_junk and
            not is_advertisement
        )

        if should_keep:
            filtered_item = item.copy()
            filtered_item['text_cleaned'] = cleaned_text
            filtered_item['answer_cleaned'] = cleaned_answer
            filtered_data.append(filtered_item)

    return filtered_data

# ===================== Главная функция =====================
def main():
    print("Загружаем wb-feedbacks...")
    
    try:
        # Пробуем загрузить с использованием num_proc для параллельной обработки
        dataset = load_dataset("nyuuzyou/wb-feedbacks", split="train", num_proc=4)
        print(f"Загружено {len(dataset)} строк")
        
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        print("Пробуем альтернативный подход...")
        
        # Альтернативный подход - скачиваем и обрабатываем локально
        try:
            # Скачиваем датасет
            dataset = load_dataset("nyuuzyou/wb-feedbacks", split="train", cache_dir="./dataset_cache")
            print(f"Загружено {len(dataset)} строк")
        except Exception as e2:
            print(f"Не удалось загрузить датасет: {e2}")
            return

    total_processed = 0
    total_filtered = 0
    chunk_files = []

    # Разбиваем на чанки и обрабатываем
    num_chunks = (len(dataset) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for chunk_idx in tqdm(range(num_chunks), desc="Обработка чанков"):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(dataset))
        
        chunk_data = dataset.select(range(start_idx, end_idx))
        total_processed += len(chunk_data)
        
        # Фильтруем чанк
        filtered_data = process_chunk(chunk_data)
        total_filtered += len(filtered_data)
        
        if len(filtered_data) > 0:
            # Сохраняем временный файл
            temp_file = f"{TEMP_DIR}/chunk_{chunk_idx}.parquet"
            filtered_df = pd.DataFrame(filtered_data)
            filtered_df.to_parquet(temp_file, index=False)
            chunk_files.append(temp_file)
        
        print(f"\nЧанк {chunk_idx + 1}: обработано {len(chunk_data)}, отфильтровано {len(filtered_data)}")
        
    
    # Объединяем все чанки
    if chunk_files:
        print("Объединяем результаты...")
        all_chunks = []
        for file in chunk_files:
            chunk_df = pd.read_parquet(file)
            all_chunks.append(chunk_df)
        
        final_df = pd.concat(all_chunks, ignore_index=True)
        
        # Сохранение финального файла
        final_df['formatted_text'] = "Отзыв: " + final_df['text_cleaned'] + "\nОтвет: " + final_df['answer_cleaned']
        final_df.to_parquet(OUTPUT_FILE, index=False)
        
        # Статистика
        print(f"\n📊 Статистика:")
        print(f"Всего обработано: {total_processed} строк")
        print(f"После фильтрации: {total_filtered} строк")
        print(f"Процент сохраненных: {total_filtered/total_processed*100:.1f}%")
        print(f"🚫 Отсеяно: {total_processed - total_filtered} строк")
        print(f"✅ Сохранено в: {OUTPUT_FILE}")
        
        # Очистка временных файлов
        for file in chunk_files:
            os.remove(file)
        os.rmdir(TEMP_DIR)
    
    else:
        print("❌ Не осталось данных после фильтрации")

if __name__ == "__main__":
    main()