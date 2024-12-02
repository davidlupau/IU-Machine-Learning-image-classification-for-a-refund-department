import os
from functions import connect_dataset, extract_combined_features, process_all_images

# Define our paths
DATA_DIR = "Dataset"
CSV_PATH = os.path.join(DATA_DIR, "styles.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

