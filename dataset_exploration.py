from functions import analyze_dataset, prepare_dataset
import os


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