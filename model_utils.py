import numpy as np
from PIL import Image
import joblib
import logging
from functions import extract_enhanced_features
import pandas as pd


class ImageProcessor:
    def __init__(self, model_path='random_forest_model.joblib'):
        """Initialize the image processor with the trained model"""
        try:
            self.model = joblib.load(model_path)
            self.logger = logging.getLogger(__name__)

            # Load feature matrix to get column names
            feature_matrix = pd.read_csv('final_feature_matrix.csv')
            # Get feature columns (excluding 'image_id' and 'category')
            self.feature_columns = [col for col in feature_matrix.columns
                                    if col not in ['image_id', 'category']]

            self.logger.info(f"Loaded model and {len(self.feature_columns)} feature columns")
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise

    def extract_features(self, image_path):
        """Extract features from an image using the same process as training"""
        try:
            # Extract features using the same function as training
            features_dict = extract_enhanced_features(image_path)

            if features_dict is None:
                raise ValueError("Feature extraction returned None")

            # Convert to DataFrame with same columns as training
            features_df = pd.DataFrame([features_dict])

            # Select only the features used in training
            features = features_df[self.feature_columns]

            # Handle infinite and missing values as in training
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(features.mean())

            # Convert to numpy array
            features_array = features.values

            self.logger.info(f"Successfully extracted {features_array.shape[1]} features")
            return features_array

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def predict(self, features):
        """Make predictions using the trained model"""
        try:
            # Get probability predictions
            probabilities = self.model.predict_proba(features)[0]

            # Create dictionary of predictions
            predictions = {
                str(class_name): float(prob)
                for class_name, prob in zip(self.model.classes_, probabilities)
            }

            # Sort by probability
            predictions = dict(sorted(predictions.items(),
                                      key=lambda item: item[1],
                                      reverse=True))

            return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise