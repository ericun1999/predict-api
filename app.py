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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Set device (Render is CPU-only)
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
        model = models.resnet18(pretrained=False)  # No pretrained weights
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Binary classification: damaged, undamaged
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        # Optional: Quantize model to reduce memory footprint
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
        # Pre-resize large images to reduce memory/CPU usage
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

# Load model at startup
model_path = 'best_model.pth'  # Ensure this file exists in your Render environment
try:
    model = load_model(model_path)
except Exception as e:
    logger.critical(f"Model loading failed, exiting: {str(e)}")
    exit(1)

# Root endpoint for health checks or basic info
@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'API is running', 'endpoint': '/predict for POST requests'})

# Health check endpoint for Render
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
    
    if file:
        try:
            img_bytes = file.read()
            log_memory_usage()  # Log memory before prediction
            prediction, confidence = predict_image(img_bytes)
            logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")
            log_memory_usage()  # Log memory after prediction
            return jsonify({
                'prediction': prediction,
                'confidence': confidence
            })
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    # 强制使用8080端口，与Cloud Run要求一致
    app.run(host='0.0.0.0', port=8080, debug=False)