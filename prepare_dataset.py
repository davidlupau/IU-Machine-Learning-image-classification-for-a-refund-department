from functions import prepare_dataset, connect_dataset, process_all_images, extract_enhanced_features
import os

def main():
    DATA_DIR = "Dataset"
    CSV_PATH = os.path.join(DATA_DIR, "styles.csv")
    IMAGES_DIR = os.path.join(DATA_DIR, "train_images")
    BACKUP_DIR = os.path.join(DATA_DIR, "excluded_items")

    print("Preparing dataset...")
    results = prepare_dataset(DATA_DIR, CSV_PATH, IMAGES_DIR, BACKUP_DIR)

    print("\nConnecting dataset...")
    valid_data, missing = connect_dataset(CSV_PATH, IMAGES_DIR)

    print("\nStarting feature extraction...")
    # Test with single image first
    test_image_path = valid_data['image_path'].iloc[0]
    test_features = extract_enhanced_features(test_image_path)

    if test_features is not None:
        print("Single image test successful!")
        print("\nProceeding with full dataset processing...")
        feature_matrix, processed_paths, failed_paths = process_all_images(valid_data)
        return valid_data, missing, feature_matrix
    else:
        print("Feature extraction test failed")
        return valid_data, missing, None

if __name__ == "__main__":
    main()