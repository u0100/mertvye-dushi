import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import requests

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://vadim-s-portfolio.vercel.app"}})  # Разрешаем доступ только с указанного домена

# Ссылки на файлы
MODEL_URL = "https://raw.githubusercontent.com/u0100/mertvye-dushi/main/text_generation_model_v2.h5"
TEXT_URL = "https://raw.githubusercontent.com/u0100/mertvye-dushi/main/mertvye-dushi.txt"

# Локальные пути для временного хранения файлов
MODEL_PATH = "/tmp/text_generation_model_v2.h5"
TEXT_PATH = "/tmp/mertvye-dushi.txt"

# Функция для скачивания файлов
def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        logging.info(f"Файл {save_path} загружен успешно!")
    else:
        logging.error(f"Ошибка загрузки файла: {url}")
        raise Exception(f"Не удалось скачать файл: {url}")

# Скачиваем файлы
if not os.path.exists(MODEL_PATH):
    download_file(MODEL_URL, MODEL_PATH)
if not os.path.exists(TEXT_PATH):
    download_file(TEXT_URL, TEXT_PATH)

# Загружаем текст книги
with open(TEXT_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# Создаём уникальный набор символов
vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Загружаем обученную модель
model = tf.keras.models.load_model(MODEL_PATH)

# Функция генерации текста
def generate_text(model, start_string, num_generate=1000, temperature=0.6):
    input_eval = [char2idx[s] for s in start_string if s in char2idx]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :]
        predictions /= temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
        if idx2char[predicted_id] in {'.', '!', '?'}:
            break

    return start_string + ''.join(text_generated)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    logging.info(f"Received request: {data}")  # Логируем входные данные

    start_string = data.get("start_string", "Чичиков произнес: ")
    num_generate = int(data.get("num_generate", 70))
    temperature = float(data.get("temperature", 0.8))

    generated_text = generate_text(model, start_string, num_generate, temperature)
    
    logging.info(f"Generated text: {generated_text}")  # Логируем выходные данные
    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
