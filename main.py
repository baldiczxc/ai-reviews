# test_rugpt3_lora.py
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline
)
from peft import PeftModel

# Конфигурация
MODEL_NAME = "ai-forever/rugpt3medium_based_on_gpt2"
LORA_PATH = "./review_responder_rugpt3_lora_filtered"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 192

def load_model():
    """Загружаем базовую модель и LoRA адаптер"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()  # Объединяем веса для более быстрого вывода
    
    return model, tokenizer

def generate_response(model, tokenizer, review, max_length=MAX_LENGTH):
    """Генерируем ответ на отзыв"""
    prompt = f"Отзыв: {review}\nОтвет:"
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length
    ).to(DEVICE)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.5,  # Уменьшаем "креативность"
        top_k=50,         # Ограничиваем словарь
        repetition_penalty=1.5,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Извлекаем только сгенерированный ответ (после "Ответ:")
    answer_start = response.find("Ответ:") + len("Ответ:")
    generated_answer = response[answer_start:].strip()
    
    return generated_answer

def interactive_test(model, tokenizer):
    """Интерактивное тестирование модели"""
    print("\n" + "="*50)
    print("Интерактивный режим тестирования модели")
    print("Введите 'exit' для выхода")
    print("="*50 + "\n")
    
    while True:
        review = input("Введите отзыв: ").strip()
        if review.lower() == 'exit':
            break
            
        if not review:
            print("Пожалуйста, введите текст отзыва")
            continue
            
        print("\nГенерируем ответ...\n")
        answer = generate_response(model, tokenizer, review)
        print(f"Ответ: {answer}\n")

if __name__ == "__main__":
    print("Загружаем модель...")
    model, tokenizer = load_model()
    model.eval()
    print("Модель загружена успешно!\n")
    
    # Тестовые примеры
    test_reviews = [
        "Товар хороший, но доставили позже чем обещали.",
        "Качество отличное, всем рекомендую!",
        "Ужасное качество, деньги на ветер.",
        "Не соответствует описанию, очень разочарован."
    ]
    
    print("Тестовые запуски:\n")
    for review in test_reviews:
        answer = generate_response(model, tokenizer, review)
        print(f"Отзыв: {review}")
        print(f"Ответ: {answer}\n")
    
    # Интерактивный режим
    interactive_test(model, tokenizer)