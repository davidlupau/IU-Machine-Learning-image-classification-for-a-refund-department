import numpy as np
from PIL import Image
import joblib
import logging
from functions import extract_enhanced_features
from scipy import stats, ndimage
from skimage import feature

class ImageProcessor:
    def __init__(self, model_path='random_forest_model.joblib'):
        self.model = joblib.load(model_path)
        self.logger = logging.getLogger(__name__)

    def extract_features(self, image):
        try:
            # Ensure image is RGB and correct size
            if isinstance(image, str):
                image = Image.open(image)
            image = image.convert('RGB')
            image = image.resize((60, 80))

            # Extract features using the enhanced function
            features_dict = extract_enhanced_features(image)

            # Convert to array
            features = list(features_dict.values())
            features = np.array(features).reshape(1, -1)

            # Clean the features - handle infinities and large values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Log the cleaned features shape
            self.logger.info(f"Cleaned feature array shape: {features.shape}")

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            raise

    def predict(self, features):
        try:
            probabilities = self.model.predict_proba(features)[0]

            predictions = {
                str(class_name): float(prob)
                for class_name, prob in zip(self.model.classes_, probabilities)
            }

            return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise