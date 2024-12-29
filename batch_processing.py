import os
import logging
import pandas as pd
from datetime import datetime
import shutil
from model_utils import ImageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='batch_processing.log'
)


class BatchProcessor:
    def __init__(self,
                 input_dir='./data/new_returns',
                 output_dir='./data/processed_returns',
                 archive_dir='./data/processed_archive'):
        """
        Initialize BatchProcessor with local directory configuration

        Args:
            input_dir (str): Directory containing new images to process
            output_dir (str): Directory to store current processing results
            archive_dir (str): Directory to archive processed images
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.archive_dir = archive_dir
        self.model_processor = ImageProcessor()  # Initialize model processor here

        # Create directories if they don't exist
        for dir_path in [input_dir, output_dir, archive_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def get_unprocessed_images(self):
        """
        Retrieve list of unprocessed images from input directory
        Returns:
            list: Paths of unprocessed image files
        """
        return [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def process_image(self, image_file):
        """
        Process a single image through the prediction API

        Args:
            image_file (str): Path of the image file to process

        Returns:
            dict: Result of the image classification
        """
        try:
            # Extract features using the image processor
            features = self.model_processor.extract_features(
                os.path.join(self.input_dir, image_file)
            )

            # Get predictions from the model
            predictions = self.model_processor.predict(features)

            # Get the highest probability category
            top_category = max(predictions.items(), key=lambda x: x[1])

            return {
                'category': top_category[0],
                'confidence': top_category[1],
                'all_predictions': predictions
            }

        except Exception as e:
            logging.error(f"Error processing {image_file}: {str(e)}")
            return None

    def process_batch(self, batch_images):
        """
        Process a batch of images

        Args:
            batch_images (list): List of image file paths to process

        Returns:
            pd.DataFrame: Results of batch processing
        """
        results = []

        for image_file in batch_images:
            prediction = self.process_image(image_file)
            if prediction:
                result = {
                    'filename': image_file,
                    'predicted_category': prediction['category'],
                    'confidence': prediction['confidence']
                }
                results.append(result)

                # Move processed image to archive
                shutil.move(
                    os.path.join(self.input_dir, image_file),
                    os.path.join(self.archive_dir, image_file)
                )
                logging.info(f"Processed and archived {image_file}")

        return pd.DataFrame(results) if results else pd.DataFrame()

    def save_results(self, results_df):
        """
        Save batch processing results to CSV

        Args:
            results_df (pd.DataFrame): Dataframe of processing results
        """
        if not results_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.output_dir,
                f'batch_results_{timestamp}.csv'
            )
            results_df.to_csv(output_file, index=False)
            logging.info(f"Results saved to {output_file}")
        else:
            logging.info("No results to save")

    def run(self):
        """
        Main method to run batch processing
        """
        logging.info("Starting batch processing")

        # Get unprocessed images
        unprocessed_images = self.get_unprocessed_images()

        if not unprocessed_images:
            logging.info("No new images to process")
            return

        # Process batch
        results = self.process_batch(unprocessed_images)

        # Save results
        self.save_results(results)

        logging.info("Batch processing completed")


if __name__ == '__main__':
    processor = BatchProcessor()
    processor.run()