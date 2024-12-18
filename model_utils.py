import numpy as np
from PIL import Image
import joblib
import logging

class ImageProcessor:
    """
    Handles all image processing and feature extraction operations.
    This class encapsulates the functionality needed in both training and production.
    """
    def __init__(self, model_path='random_forest_model.joblib'):
        """Initialize the processor with a trained model."""
        self.model = joblib.load(model_path)
        self.logger = logging.getLogger(__name__)

    def extract_features(self, image):
        """
        Extract features from an image using the same process as training.

        Args:
            image (PIL.Image): Image to process

        Returns:
            np.array: Extracted features
        """
        try:
            # Import your feature extraction functions from functions.py
            from functions import extract_combined_features

            # Convert PIL image to the format your function expects
            # Adjust this based on your existing feature extraction code
            features = extract_combined_features(image)

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def predict(self, features):
        """
        Make predictions using the loaded model.

        Args:
            features (np.array): Extracted features

        Returns:
            dict: Prediction probabilities for each class
        """
        try:
            # Get probability predictions
            probabilities = self.model.predict_proba([features])[0]

            # Create a dictionary mapping classes to probabilities
            predictions = {
                str(class_name): float(prob)
                for class_name, prob in zip(self.model.classes_, probabilities)
            }

            return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise