import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import requests
from supabase import create_client, Client
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://mertvye-dushi.vercel.app", "http://localhost:3000"]}})

# Получаем переменные окружения
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Инициализация Supabase клиента
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Локальные пути для временного хранения файлов
MODEL_PATH = "/tmp/text_generation_model_v2.h5"
TEXT_PATH = "/tmp/mertvye-dushi.txt"

# Функция для скачивания файлов из Supabase Storage
def download_file_from_supabase(file_name, save_path):
    try:
        res = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)
        file_url = res
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            logging.info(f"Файл {file_name} загружен успешно!")
        else:
            logging.error(f"Ошибка загрузки файла: {file_name}")
            raise Exception(f"Не удалось скачать файл: {file_name}")
    except Exception as e:
        logging.error(f"Ошибка при скачивании файла: {e}")
        raise

# Скачиваем файлы, если они еще не загружены
if not os.path.exists(MODEL_PATH):
    download_file_from_supabase("text_generation_model_v2.h5", MODEL_PATH)
if not os.path.exists(TEXT_PATH):
    download_file_from_supabase("mertvye-dushi.txt", TEXT_PATH)

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

@app.route('/')
def home():
    return "Welcome to the Text Generation API!"

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        logging.info(f"Received request: {data}")

        start_string = data.get("start_string", "Чичиков произнес: ")
        num_generate = int(data.get("num_generate", 70))
        temperature = float(data.get("temperature", 0.8))

        generated_text = generate_text(model, start_string, num_generate, temperature)
        
        logging.info(f"Generated text: {generated_text}")
        return jsonify({"generated_text": generated_text})
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
