from functions import analyze_dataset, prepare_dataset, connect_dataset
import os
from functions import connect_dataset, extract_combined_features, process_all_images

# Set paths
DATA_DIR = "Dataset"
CSV_PATH = os.path.join(DATA_DIR, "styles.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
BACKUP_DIR = os.path.join(DATA_DIR, "excluded_items")

# Explore the dataset and images directory to get an overview
analyze_dataset(CSV_PATH, IMAGES_DIR)

if __name__ == "__main__":
    print("This script will:")
    print("1. Keep only essential columns (id, masterCategory, gender, baseColour)")
    print("2. Filter out small categories")
    print("3. Move excluded images to backup")
    print("4. Create a backup of the original CSV")
    response = input("\nDo you want to proceed? (yes/no): ")

    if response.lower() == 'yes':
        # Get all results in one dictionary
        results = prepare_dataset(DATA_DIR, CSV_PATH, IMAGES_DIR, BACKUP_DIR)

        # Print results
        print("\nDataset preparation completed:")
        print(f"Original dataset shape: {results['original_shape']}")
        print(f"Filtered dataset shape: {results['filtered_shape']}")
        print("\nKept columns:", results['kept_columns'])

        print("\nFinal category distribution:")
        print(results['category_distribution'])

        print("\nMissing values in kept columns:")
        print(results['missing_values'])

        print(f"\nImages moved to backup: {results['moved_count']}")

        # Check for inconsistencies
        if results['inconsistencies']['missing_images'] or results['inconsistencies']['extra_images']:
            print("\nWarning: Inconsistencies found:")
            if results['inconsistencies']['missing_images']:
                print(f"Missing images: {len(results['inconsistencies']['missing_images'])}")
            if results['inconsistencies']['extra_images']:
                print(f"Extra images: {len(results['inconsistencies']['extra_images'])}")
        else:
            print("\nDataset is consistent: all CSV entries have corresponding images")
    else:
        print("Operation cancelled")

# Load CSV metadata and connect with image files
if __name__ == "__main__":
    # Connect dataset
    valid_data, missing = connect_dataset(CSV_PATH, IMAGES_DIR)

    # Show sample of the data
    print("\nSample of valid data:")
    print(valid_data[['id', 'masterCategory', 'image_path']].head())

    # Show distribution of categories
    print("\nCategory distribution:")
    print(valid_data['masterCategory'].value_counts())

# Feature extraction
if __name__ == "__main__":
    # First, test with a single image to ensure everything works
    valid_data, missing = connect_dataset(CSV_PATH, IMAGES_DIR)

    print("Testing with a single image first...")
    test_image_path = valid_data['image_path'].iloc[0]
    test_features = extract_combined_features(test_image_path)

    if test_features is not None:
        print("Single image test successful!")
        print(f"Features extracted: {list(test_features.keys())}")

        # If single image test succeeds, process all images
        print("\nProceeding with full dataset processing...")
        feature_matrix, processed_paths, failed_paths = process_all_images(valid_data)

        print("\nFeature Matrix Information:")
        print(feature_matrix.info())
        print("\nSample of feature matrix:")
        print(feature_matrix.head())
    else:
        print("Single image test failed. Please check the error messages above.")