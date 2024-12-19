from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import logging
from datetime import datetime
from model_utils import ImageProcessor

# Initialize Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize the image processor
try:
    processor = ImageProcessor()
    logging.info("Model and processor initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize model: {str(e)}")
    raise

@app.route('/health', methods=['GET'])
def health_check():
    """Simple endpoint to verify the API is running."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Endpoint for single image prediction.
    Expects JSON with base64 encoded image.
    """
    try:
        # Get image data from request
        if not request.json or 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400

        # Decode and load image
        image_data = base64.b64decode(request.json['image'])
        image = Image.open(io.BytesIO(image_data))

        # Extract features and predict
        features = processor.extract_features(image)
        predictions = processor.predict(features)

        # Create response
        response = {
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }

        logging.info("Successfully processed single prediction")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Endpoint for batch prediction.
    Expects JSON with list of base64 encoded images.
    """
    try:
        # Validate request
        if not request.json or 'images' not in request.json:
            return jsonify({'error': 'No images provided'}), 400

        # Process each image
        results = []
        for image_data in request.json['images']:
            # Decode and process each image
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            features = processor.extract_features(image)
            predictions = processor.predict(features)
            results.append(predictions)

        # Create response
        response = {
            'predictions': results,
            'batch_size': len(results),
            'timestamp': datetime.now().isoformat()
        }

        logging.info(f"Successfully processed batch of {len(results)} images")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error processing batch request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)