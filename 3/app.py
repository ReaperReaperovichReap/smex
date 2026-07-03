import os
import pickle
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, render_template, jsonify
import io

# ---------- Инициализация Flask ----------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 МБ лимит

# ---------- Загрузка модели и классов ----------
base_dir = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(base_dir, 'classes.pkl'), 'rb') as f:
    CLASS_NAMES = pickle.load(f)
NUM_CLASSES = len(CLASS_NAMES)

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)
model.load_state_dict(torch.load(os.path.join(base_dir, 'smeshariki_classifier.pth'), map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

IMAGE_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image_bytes):
    """Принимает байты изображения, возвращает словарь с топ-3 предсказаниями."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, 3)
    result = {}
    for prob, idx in zip(top_probs, top_indices):
        class_name = CLASS_NAMES[idx.item()]
        confidence = prob.item() * 100
        result[class_name] = confidence
    return result

# ---------- Маршруты ----------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    try:
        image_bytes = file.read()
        predictions = predict_image(image_bytes)
        # Формируем ответ в виде JSON
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)