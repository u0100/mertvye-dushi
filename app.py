from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://vadim-s-portfolio.vercel.app"}})  # Разрешаем доступ только с указанного домена

# Пути к файлам
MODEL_PATH = "./text_generation_model.h5"
TEXT_PATH = "./mertvye-dushi.txt"

# Загружаем текст книги
with open(TEXT_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# Создаём уникальный набор символов
vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Загружаем обученную модель
model = tf.keras.models.load_model(MODEL_PATH)

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
    start_string = data.get("start_string", "Чичиков произнес: ")
    num_generate = int(data.get("num_generate", 70))
    temperature = float(data.get("temperature", 0.8))
    
    generated_text = generate_text(model, start_string, num_generate, temperature)
    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
