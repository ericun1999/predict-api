import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from flask import Flask, request, jsonify
import os
import logging
import psutil
import time

# Configure logging for Heroku
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Set device (Heroku is CPU-only)
device = torch.device("cpu")
logger.info(f"Using device: {device}")

# Log memory usage
def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage: RSS={mem_info.rss / 1024**2:.2f} MB, VMS={mem_info.vms / 1024**2:.2f} MB")

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class names
class_names = ['damaged', 'undamaged']

# Load ResNet18 model
def load_model(model_path):
    try:
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        model = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
        logger.info("Model loaded successfully")
        log_memory_usage()
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

# Prediction function
def predict_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if image.size[0] > 1000 or image.size[1] > 1000:
            image = image.resize((1000, 1000))
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            prob = torch.softmax(outputs, dim=1)[0][predicted].item()
        
        return class_names[predicted], prob
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

# Lazy-load model
model = None
def load_model_if_needed():
    global model
    if model is None:
        model_path = 'best_model.pth'
        model = load_model(model_path)
    return model

# Root endpoint
@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'API is running', 'endpoint': '/predict for POST requests'})

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    logger.info("Received prediction request")
    
    if 'file' not in request.files:
        logger.error("No file provided")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file size (max 10 MB)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    if file_size > 10 * 1024 * 1024:
        logger.error("File too large")
        return jsonify({'error': 'File too large, max 10 MB'}), 400
    file.seek(0)
    
    try:
        img_bytes = file.read()
        log_memory_usage()
        load_model_if_needed()  # Lazy-load model
        prediction, confidence = predict_image(img_bytes)
        logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")
        log_memory_usage()
        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)