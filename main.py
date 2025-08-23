from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def quick_test():
    model_path = "./t5_scratch_run/model"
    
    # Загрузка модели
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Тестовый отзыв
    review = "Товар пришел с браком, очень расстроен качеством."
    input_text = "Отзыв: " + review
    
    # Токенизация
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)
    
    # Генерация
    outputs = model.generate(
        input_ids,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    
    # Декодирование
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Отзыв: {review}")
    print(f"Ответ: {response}")

if __name__ == "__main__":
    quick_test()