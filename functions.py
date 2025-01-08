import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
from skimage import feature, filters
from scipy import ndimage, stats
import numpy as np
from PIL import Image
from scipy import stats, ndimage
from skimage import feature, measure
from skimage.feature import graycomatrix, graycoprops
import cv2

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

    # Categories
    categories = df['categories'].value_counts()
    print("\nCategories:")
    print(categories)
    print(f"Total number of categories: {len(categories)}")

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("\n4. MISSING VALUES")
        print("-" * 50)
        print(missing_values[missing_values > 0])

    # Analyze a small sample of images for size information
    print("\n5. IMAGE SAMPLE ANALYSIS")
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
def prepare_dataset(data_dir, CSV_PATH, images_dir, backup_dir):
    """
    Prepare dataset by cleaning and organizing data
    """
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    # Print initial shape
    print(f"\nInitial dataset shape: {df.shape}")
    print("Initial columns:", df.columns.tolist())

    # Keep only essential columns
    keep_columns = ['ImgId', 'categories']
    df_filtered = df[keep_columns]

    print("\nAfter filtering:")
    print(f"Final dataset shape: {df_filtered.shape}")
    print("\nCategory distribution:")
    print(df_filtered['categories'].value_counts())

    # Save CSVs
    backup_CSV_PATH = os.path.join(backup_dir, 'styles_original.csv')
    df.to_csv(backup_CSV_PATH, index=False)  # Backup
    df_filtered.to_csv(CSV_PATH, index=False)  # Filtered
    print(f"\nSaved backup to: {backup_CSV_PATH}")
    print(f"Saved filtered data to: {CSV_PATH}")

    # Return dictionary with all results
    results = {
        'original_shape': df.shape,
        'filtered_shape': df_filtered.shape,
        'kept_columns': df_filtered.columns.tolist(),
        'category_distribution': df_filtered['categories'].value_counts(),
        'missing_values': df_filtered.isnull().sum()
    }

    # Add consistency check results
    print("\nPerforming consistency checks...")
    remaining_images = set(os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith('.jpg'))
    csv_ids = set(df_filtered['ImgId'].astype(str))

    results['inconsistencies'] = {
        'missing_images': csv_ids - remaining_images,
        'extra_images': remaining_images - csv_ids
    }

    print(f"Missing images: {len(results['inconsistencies']['missing_images'])}")
    print(f"Extra images: {len(results['inconsistencies']['extra_images'])}")

    return results

def connect_dataset(CSV_PATH, IMAGES_DIR):
    """
    Connect the image dataset by linking the CSV metadata with actual image files.

    Parameters:
    CSV_PATH (str): Path to the CSV file containing image metadata
    IMAGES_DIR (str): Path to the folder containing image files

    Returns:
    tuple: (DataFrame with verified image paths, list of any missing images)
    """
    print("Reading CSV metadata...")
    metadata_df = pd.read_csv(CSV_PATH)

    metadata_df['image_path'] = metadata_df['ImgId'].apply(
        lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg")
    )

    print("Verifying image files...")
    missing_images = []
    existing_images = []

    for idx, row in metadata_df.iterrows():
        if idx % 1000 == 0:
            print(f"Checking image {idx} of {len(metadata_df)}")

        if os.path.exists(row['image_path']):
            try:
                with Image.open(row['image_path']) as img:
                    existing_images.append(True)
            except Exception as e:
                print(f"Error with image {row['ImgId']}: {str(e)}")
                existing_images.append(False)
                missing_images.append(row['ImgId'])
        else:
            existing_images.append(False)
            missing_images.append(row['ImgId'])

    metadata_df['image_exists'] = existing_images
    valid_df = metadata_df[metadata_df['image_exists']][['ImgId', 'categories', 'image_path']].copy()

    print("\nDataset Summary:")
    print(f"Total entries in CSV: {len(metadata_df)}")
    print(f"Valid images found: {len(valid_df)}")
    print(f"Missing images: {len(missing_images)}")

    return valid_df, missing_images

def extract_enhanced_features(image):
   """Enhanced feature extraction with computer vision techniques"""
   try:
       # Load and prepare image
       if isinstance(image, str):
           image = Image.open(image)
       image = image.convert('RGB')
       image = image.resize((60, 80))
       img_array = np.array(image)
       gray_image = np.mean(img_array, axis=2).astype(np.uint8)

       # 1. Shape Features
       shape_features = {
           'aspect_ratio': img_array.shape[0] / img_array.shape[1],
           'vertical_symmetry': np.mean(np.abs(gray_image - np.flipud(gray_image))),
           'horizontal_symmetry': np.mean(np.abs(gray_image - np.fliplr(gray_image))),
           'diagonal_symmetry': np.mean(np.abs(gray_image - np.rot90(np.rot90(gray_image)))),
           'quadrant_symmetry': np.mean([np.abs(gray_image[:gray_image.shape[0]//2, :gray_image.shape[1]//2] -
                                       gray_image[gray_image.shape[0]//2:, gray_image.shape[1]//2:])])
       }

       # Add contour features
       contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       if contours:
           main_contour = max(contours, key=cv2.contourArea)
           shape_features.update({
               'contour_area': cv2.contourArea(main_contour),
               'contour_perimeter': cv2.arcLength(main_contour, True),
               'contour_circularity': (4 * np.pi * cv2.contourArea(main_contour)) /
                                    (cv2.arcLength(main_contour, True) ** 2) if cv2.arcLength(main_contour, True) > 0 else 0
           })

       # 2. Texture Features
       lbp = feature.local_binary_pattern(gray_image, P=8, R=1)
       texture_features = {
           'texture_mean': lbp.mean(),
           'texture_var': lbp.var(),
           'texture_uniformity': len(np.unique(lbp)) / len(lbp.flatten())
       }

       # GLCM features
       angles = [0, 45, 90, 135]
       glcm = graycomatrix(gray_image, distances=[1, 2], angles=angles, normed=True)
       for angle_idx, angle in enumerate(angles):
           texture_features.update({
               f'glcm_contrast_{angle}': graycoprops(glcm, 'contrast')[0, angle_idx],
               f'glcm_homogeneity_{angle}': graycoprops(glcm, 'homogeneity')[0, angle_idx],
               f'glcm_energy_{angle}': graycoprops(glcm, 'energy')[0, angle_idx],
               f'glcm_correlation_{angle}': graycoprops(glcm, 'correlation')[0, angle_idx]
           })

       # 3. Edge Features
       sobel_h = ndimage.sobel(gray_image, axis=0)
       sobel_v = ndimage.sobel(gray_image, axis=1)
       edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
       canny_edges = feature.canny(gray_image, sigma=1.0)

       edge_features = {
           'edge_density': np.mean(edge_magnitude),
           'edge_variance': np.var(edge_magnitude),
           'horizontal_edges': np.mean(np.abs(sobel_h)),
           'vertical_edges': np.mean(np.abs(sobel_v)),
           'canny_edge_density': np.mean(canny_edges),
           'edge_magnitude_mean': np.mean(edge_magnitude),
           'edge_magnitude_std': np.std(edge_magnitude)
       }

       # Edge direction histogram
       edge_angles = np.arctan2(sobel_v, sobel_h) * 180 / np.pi
       hist, _ = np.histogram(edge_angles[edge_magnitude > edge_magnitude.mean()],
                            bins=12, range=(-180, 180))
       for i, count in enumerate(hist):
           edge_features[f'edge_direction_bin_{i}'] = count
       edge_features['edge_direction_entropy'] = stats.entropy(hist + 1)

       # 4. Color Features
       color_features = {}

       # RGB features
       for idx, channel in enumerate(['red', 'green', 'blue']):
           channel_data = img_array[:,:,idx]
           color_features.update({
               f'mean_{channel}': channel_data.mean(),
               f'std_{channel}': channel_data.std(),
               f'skew_{channel}': stats.skew(channel_data.flatten()),
               f'kurtosis_{channel}': stats.kurtosis(channel_data.flatten())
           })

       # HSV features
       hsv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
       for idx, channel in enumerate(['hue', 'saturation', 'value']):
           channel_data = hsv_image[:,:,idx]
           color_features.update({
               f'mean_{channel}': channel_data.mean(),
               f'std_{channel}': channel_data.std(),
               f'kurtosis_{channel}': stats.kurtosis(channel_data.flatten())
           })

       # LAB color space features
       lab_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
       for idx, channel in enumerate(['l', 'a', 'b']):
           channel_data = lab_image[:,:,idx]
           color_features.update({
               f'mean_{channel}': np.mean(channel_data),
               f'std_{channel}': np.std(channel_data),
               f'kurtosis_{channel}': stats.kurtosis(channel_data.flatten())
           })

       # Color histogram features
       for idx, channel in enumerate(['red', 'green', 'blue']):
           hist, _ = np.histogram(img_array[:,:,idx], bins=8, range=(0, 256))
           for bin_idx, count in enumerate(hist):
               color_features[f'{channel}_hist_bin_{bin_idx}'] = count

       return {**shape_features, **texture_features, **edge_features, **color_features}

   except Exception as e:
       print(f"Error in feature extraction: {str(e)}")
       return None

def process_all_images(valid_data):
   """Process all images and create feature matrix"""
   image_paths = valid_data['image_path'].tolist()
   print(f"Starting to process {len(image_paths)} images...")

   features_list = []
   processed_paths = []
   failed_paths = []

   for idx, image_path in enumerate(tqdm(image_paths, desc="Extracting features")):
       try:
           features = extract_enhanced_features(image_path)

           if features is not None:
               # Add image ID and category
               image_id = os.path.basename(image_path).split('.')[0]
               features['image_id'] = image_id
               features['category'] = valid_data.loc[valid_data['image_path'] == image_path, 'categories'].iloc[0]

               features_list.append(features)
               processed_paths.append(image_path)
           else:
               failed_paths.append(image_path)

           # Checkpoint every 1000 images
           if (idx + 1) % 1000 == 0:
               print(f"\nCheckpoint: Processed {idx + 1} images")
               pd.DataFrame(features_list).to_csv(f'features_checkpoint_{idx + 1}.csv', index=False)

       except Exception as e:
           print(f"\nError processing {image_path}: {str(e)}")
           failed_paths.append(image_path)

   # Save results
   feature_matrix = pd.DataFrame(features_list)
   feature_matrix.to_csv('final_feature_matrix.csv', index=False)

   if failed_paths:
       with open('failed_images.txt', 'w') as f:
           f.write('\n'.join(failed_paths))

   print("\nProcessing Complete!")
   print(f"Successfully processed: {len(processed_paths)} images")
   print(f"Failed to process: {len(failed_paths)} images")
   print(f"Total features per image: {len(feature_matrix.columns) - 2}")

   return feature_matrix, processed_paths, failed_paths
