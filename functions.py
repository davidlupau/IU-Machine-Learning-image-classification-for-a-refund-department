import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

# Function to get the dataset information
def analyze_dataset(CSV_PATH, IMAGES_DIR):
    """Analyze both metadata and images"""
    print("\nLoading and analyzing dataset...")

    # Load metadata
    df = pd.read_csv(CSV_PATH)
    print("\n1. BASIC DATASET INFORMATION")
    print("-" * 50)
    print(f"Total number of entries in CSV: {len(df)}")

    # Count images
    image_files = list(Path(IMAGES_DIR).glob("*.jpg"))
    print(f"Total number of images found: {len(image_files)}")

    # Category analysis
    print("\n2. CATEGORY ANALYSIS")
    print("-" * 50)

    # Master categories
    master_cats = df['masterCategory'].value_counts()
    print("\nMaster Categories:")
    print(master_cats)
    print(f"Total number of master categories: {len(master_cats)}")

    # Subcategories
    sub_cats = df['subCategory'].value_counts()
    print("\nSubcategories:")
    print(sub_cats)
    print(f"Total number of subcategories: {len(sub_cats)}")

    # Article types
    article_types = df['articleType'].value_counts()
    print("\nArticle Types (first 10):")
    print(article_types.head(10))
    print(f"Total number of article types: {len(article_types)}")

    # Colors
    colors = df['baseColour'].value_counts()
    print("\nColors:")
    print(colors)
    print(f"Total number of colors: {len(colors)}")

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("\n3. MISSING VALUES")
        print("-" * 50)
        print(missing_values[missing_values > 0])

    # Analyze a small sample of images for size information
    print("\n4. IMAGE SAMPLE ANALYSIS")
    print("-" * 50)
    sample_size = 50
    sample_files = np.random.choice(image_files, min(sample_size, len(image_files)), replace=False)

    file_sizes = []
    dimensions = []

    for img_path in tqdm(sample_files, desc="Analyzing sample images"):
        try:
            # Get file size in KB
            file_sizes.append(os.path.getsize(img_path) / 1024)

            # Get dimensions
            with Image.open(img_path) as img:
                width, height = img.size
                dimensions.append((width, height))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if file_sizes and dimensions:
        print(f"\nAverage file size: {np.mean(file_sizes):.2f} KB")
        print(f"Average dimensions: {np.mean([d[0] for d in dimensions]):.0f} x {np.mean([d[1] for d in dimensions]):.0f}")

# Function to prepare the dataset by filtering categories and columns
def prepare_dataset(data_dir, csv_path, images_dir, backup_dir):
    """
    Prepare dataset by filtering categories and columns
    """
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    # Print initial shape
    print(f"\nInitial dataset shape: {df.shape}")
    print("Initial columns:", df.columns.tolist())

    # Define categories and columns to keep
    keep_categories = ['Apparel', 'Accessories', 'Footwear', 'Personal Care']
    keep_columns = ['id', 'masterCategory', 'gender', 'baseColour']

    # Filter rows and columns
    df_filtered = df[df['masterCategory'].isin(keep_categories)][keep_columns]

    # Get list of images to move
    images_to_move = df[~df['masterCategory'].isin(keep_categories)]['id'].astype(str) + '.jpg'

    # Move images
    print(f"\nMoving {len(images_to_move)} images to backup...")
    moved_count = 0
    for img_name in tqdm(images_to_move):
        src_path = os.path.join(images_dir, img_name)
        dst_path = os.path.join(backup_dir, img_name)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            moved_count += 1

    # Save CSVs
    backup_csv_path = os.path.join(backup_dir, 'styles_original.csv')
    df.to_csv(backup_csv_path, index=False)  # Backup
    df_filtered.to_csv(csv_path, index=False)  # Filtered

    # Return dictionary with all results
    results = {
        'original_shape': df.shape,
        'filtered_shape': df_filtered.shape,
        'kept_columns': df_filtered.columns.tolist(),
        'category_distribution': df_filtered['masterCategory'].value_counts(),
        'missing_values': df_filtered.isnull().sum(),
        'moved_count': moved_count
    }

    # Add consistency check results
    remaining_images = set(os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith('.jpg'))
    csv_ids = set(df_filtered['id'].astype(str))

    results['inconsistencies'] = {
        'missing_images': csv_ids - remaining_images,
        'extra_images': remaining_images - csv_ids
    }

    return results