import numpy as np
from PIL import Image
import joblib
import logging
from functions import extract_combined_features
from scipy import stats, ndimage
from skimage import feature

class ImageProcessor:
    def __init__(self, model_path='random_forest_model.joblib'):
        self.model = joblib.load(model_path)
        self.logger = logging.getLogger(__name__)

    def extract_features(self, image):
        try:
            # Instead of using the original function directly,
            # we'll implement the feature extraction here

            # Ensure image is RGB and correct size
            image = image.convert('RGB')
            image = image.resize((60, 80))
            img_array = np.array(image)
            gray_image = np.mean(img_array, axis=2)

            # Extract shape features
            shape_features = {
                'aspect_ratio': img_array.shape[0] / img_array.shape[1],
                'vertical_symmetry': np.mean(np.abs(gray_image - np.flipud(gray_image))),
                'horizontal_symmetry': np.mean(np.abs(gray_image - np.fliplr(gray_image)))
            }

            # Extract texture features
            lbp = feature.local_binary_pattern(gray_image, P=8, R=1)
            texture_features = {
                'texture_mean': lbp.mean(),
                'texture_var': lbp.var(),
                'texture_uniformity': len(np.unique(lbp)) / len(lbp.flatten())
            }

            # Extract edge features
            sobel_h = ndimage.sobel(gray_image, axis=0)
            sobel_v = ndimage.sobel(gray_image, axis=1)
            edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
            canny_edges = feature.canny(gray_image, sigma=1.0)

            edge_features = {
                'edge_density': np.mean(edge_magnitude),
                'edge_variance': np.var(edge_magnitude),
                'horizontal_edges': np.mean(np.abs(sobel_h)),
                'vertical_edges': np.mean(np.abs(sobel_v)),
                'canny_edge_density': np.mean(canny_edges)
            }

            # Extract color features
            color_features = {}
            for idx, channel in enumerate(['red', 'green', 'blue']):
                channel_data = img_array[:,:,idx]
                color_features.update({
                    f'mean_{channel}': channel_data.mean(),
                    f'std_{channel}': channel_data.std(),
                    f'skew_{channel}': stats.skew(channel_data.flatten())
                })

            # Add color ratios
            color_features.update({
                'red_green_ratio': color_features['mean_red'] / (color_features['mean_green'] + 1e-6),
                'blue_green_ratio': color_features['mean_blue'] / (color_features['mean_green'] + 1e-6),
                'color_variance': np.var([color_features['mean_red'],
                                        color_features['mean_green'],
                                        color_features['mean_blue']])
            })

            # Combine all features
            features_dict = {**shape_features, **texture_features, **edge_features, **color_features}

            # Convert to array and reshape
            features = list(features_dict.values())
            features = np.array(features).reshape(1, -1)

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
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