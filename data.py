import pandas as pd
import re
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# ===================== Настройки =====================
OUTPUT_FILE = "filtered_wb_feedbacks.parquet"
MAX_ROWS = 192000000

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

# ===================== Умная фильтрация =====================
def smart_filtering(df):
    """Умная фильтрация, которая убирает рекламу и шаблонные ответы"""
    print("Запуск умной фильтрации...")

    results = []
    cleaned_texts = []
    cleaned_answers = []

    # Расширенный список стоп-слов для рекламы
    ad_phrases = [
        # Прямые призывы
        'ждем вас снова', 'ждём вас снова',
        'ждем за покупками', 'ждём за покупками',
        'приятных покупок', 'надеемся увидеть вас снова',
        'с надеждой видеть', 'будем рады видеть',
        'ждем в нашем магазине', 'ждём в нашем магазине',
        'покупай', 'заказывай', 'приобретай',
        
        # Рекламные маркеры
        'скидк', 'выгодн', 'акци', 'специальное предложение',
        'лучшее предложение', 'успейте заказать', 'успейте купить',
        'подарок каждому', 'каталог', 'ассортимент',
        
        # Замаскированные советы
        'рекомендуем попробовать', 'советуем попробовать',
        'обратите внимание на', 'посмотрите также',
        
        # Продавец сам себя хвалит
        'наши сотрудники', 'наши подруги', 'мы уверены что',
        'уверены что вы сможете подобрать'
    ]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Умная фильтрация"):
        original_text = str(row['text'])
        original_answer = str(row['answer'])

        # Очищаем от персональных данных
        cleaned_text = remove_personal_data(original_text)
        cleaned_answer = remove_personal_data(original_answer)

        # Пропускаем пустые
        if len(cleaned_text) < 10 or len(cleaned_answer) < 10:
            results.append(False)
            cleaned_texts.append(cleaned_text)
            cleaned_answers.append(cleaned_answer)
            continue

        answer_lower = cleaned_answer.lower()

        # 1. ЖЕСТКИЕ КРИТЕРИИ ОТСЕИВАНИЯ (реклама, шаблоны)
        is_advertisement = (
            re.search(r'арт\.\s*\d+|артикул\s*\d+', answer_lower) or
            re.search(r'рекомендуем.*\d+|предлагаем.*\d+', answer_lower) or
            re.search(r'[#№]\d+', answer_lower) or
            any(phrase in answer_lower for phrase in ad_phrases)
        )
        is_template = (
            len(cleaned_answer) < 25 or
            answer_lower.startswith(('спасибо', 'благодар', 'извин')) and len(cleaned_answer) < 50 or
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
            len(cleaned_answer.split()) > 8 and  # Не менее 9 слов
            any(char in cleaned_answer for char in '.!?') and  # Есть пунктуация
            any(word in answer_lower for word in [  # Содержит полезные слова
                'качеств', 'доставк', 'размер', 'цвет', 'материал',
                'гаранти', 'обмен', 'возврат', 'рекоменд', 'совету',
                'проблем', 'решен', 'изменен', 'улучшен', 'исправлен'
            ]) and
            # НЕ содержит рекламных фраз
            not any(ad_phrase in answer_lower for ad_phrase in ['купит', 'заказ', 'покупк', 'магазин', 'ассортимент'])
        )

        # 3. ФИНАЛЬНОЕ РЕШЕНИЕ - жестко отсеиваем рекламу
        should_keep = (
            has_relevance and
            has_quality and
            not is_template and
            not has_junk and
            not is_advertisement
        )

        results.append(should_keep)
        cleaned_texts.append(cleaned_text)
        cleaned_answers.append(cleaned_answer)

    # Создаем новый DataFrame с очищенными данными
    filtered_df = df.copy()
    filtered_df['text_cleaned'] = cleaned_texts
    filtered_df['answer_cleaned'] = cleaned_answers
    filtered_df = filtered_df[results]

    return filtered_df

# ===================== Показ реальных примеров =====================
def show_real_examples(original_df, filtered_df, num_examples=5):
    """Показывает реальные примеры до и после"""
    print(f"\n=== РЕАЛЬНЫЕ ПРИМЕРЫ (из {len(filtered_df)} отфильтрованных) ===")
    
    if len(filtered_df) > 0:
        sample_indices = np.random.choice(len(filtered_df), min(num_examples, len(filtered_df)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            filtered_row = filtered_df.iloc[idx]
            original_idx = filtered_row.name
            
            if original_idx in original_df.index:
                original_row = original_df.loc[original_idx]
                
                print(f"\n--- Пример {i+1} ---")
                print(f"📝 БЫЛО - Отзыв: {original_row['text']}")
                print(f"💬 БЫЛО - Ответ: {original_row['answer']}")
                print(f"✨ СТАЛО - Отзыв: {filtered_row['text_cleaned']}")
                print(f"✅ СТАЛО - Ответ: {filtered_row['answer_cleaned']}")
                
                # Анализ почему сохранили/отсеяли
                answer_lower = filtered_row['answer_cleaned'].lower()
                if any(word in answer_lower for word in ['качеств', 'гаранти', 'рекоменд']):
                    print("🔍 Сохранен: содержит полезную информацию о качестве")
                elif any(word in answer_lower for word in ['проблем', 'решен', 'исправлен']):
                    print("🔍 Сохранен: описывает решение проблемы")
                elif len(filtered_row['answer_cleaned'].split()) > 12:
                    print("🔍 Сохранен: развернутый содержательный ответ")
                else:
                    print("🔍 Сохранен: релевантный ответ без рекламы")
                
                print("-" * 80)

# ===================== Главная функция =====================
def main():
    print("Загружаем wb-feedbacks...")
    
    # Загрузка данных
    wb = load_dataset("nyuuzyou/wb-feedbacks", split="train")
    wb = wb.filter(lambda x: (x.get("text") or "") != "" and (x.get("answer") or "") != "")
    
    if len(wb) > MAX_ROWS:
        wb = wb.select(range(MAX_ROWS))
    
    original_df = pd.DataFrame(wb)
    print(f"Загружено {len(original_df)} строк")
    
    # Умная фильтрация
    filtered_df = smart_filtering(original_df)
    print(f"После фильтрации: {len(filtered_df)} строк")
    
    # Показываем реальные примеры
    show_real_examples(original_df, filtered_df)
    
    if len(filtered_df) > 0:
        # Сохранение
        filtered_df['formatted_text'] = "Отзыв: " + filtered_df['text_cleaned'] + "\nОтвет: " + filtered_df['answer_cleaned']
        filtered_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"\n✅ Сохранено {len(filtered_df)} качественных строк в {OUTPUT_FILE}")
        
        # Статистика
        print(f"\n📊 Статистика:")
        print(f"Исходно: {len(original_df)} строк")
        print(f"После фильтрации: {len(filtered_df)} строк")
        print(f"Процент сохраненных: {len(filtered_df)/len(original_df)*100:.1f}%")
        print(f"🚫 Отсеяно рекламы и шаблонов: {len(original_df) - len(filtered_df)} строк")
    
    else:
        print("❌ Не осталось данных после фильтрации")

if __name__ == "__main__":
    main()