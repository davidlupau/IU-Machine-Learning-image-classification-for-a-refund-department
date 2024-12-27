import os
import logging
import pandas as pd
import requests
from datetime import datetime
import shutil

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
                 archive_dir='./data/processed_archive',
                 api_endpoint='http://localhost:5000/predict'):
        """
        Initialize BatchProcessor with local directory configuration
        
        Args:
            input_dir (str): Directory containing new images to process
            output_dir (str): Directory to store current processing results
            archive_dir (str): Directory to archive processed images
            api_endpoint (str): URL of prediction API
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.archive_dir = archive_dir
        self.api_endpoint = api_endpoint
        
        # Create directories if they don't exist
        for dir_path in [input_dir, output_dir, archive_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
    def get_unprocessed_images(self):
        """
        Retrieve list of unprocessed images from input directory
        
        Returns:
            list: Paths of unprocessed image files
        """
        all_images = [
            f for f in os.listdir(self.input_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
        ]
        return all_images
    
    def process_batch(self, batch_images):
        """
        Process a batch of images through the prediction API
        
        Args:
            batch_images (list): List of image file paths to process
        
        Returns:
            pd.DataFrame: Results of batch processing
        """
        results = []
        
        for image_file in batch_images:
            try:
                # Read image file
                with open(os.path.join(self.input_dir, image_file), 'rb') as f:
                    files = {'file': (image_file, f)}
                    
                    # Send to prediction API
                    response = requests.post(
                        self.api_endpoint, 
                        files=files
                    )
                
                # Check response
                if response.status_code == 200:
                    prediction = response.json()
                    result = {
                        'filename': image_file,
                        'predicted_category': prediction['category'],
                        'confidence': prediction['confidence']
                    }
                    results.append(result)
                    logging.info(f"Processed {image_file}: {result}")
                    
                    # Move processed image to archive
                    shutil.move(
                        os.path.join(self.input_dir, image_file),
                        os.path.join(self.archive_dir, image_file)
                    )
                else:
                    logging.error(f"Failed to process {image_file}: {response.text}")
            
            except Exception as e:
                logging.error(f"Error processing {image_file}: {str(e)}")
        
        return pd.DataFrame(results)
    
    def save_results(self, results_df):
        """
        Save batch processing results to CSV
        
        Args:
            results_df (pd.DataFrame): Dataframe of processing results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.output_dir, 
            f'batch_results_{timestamp}.csv'
        )
        results_df.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")
    
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
