from functions import prepare_dataset, connect_dataset
import os


def main():
    DATA_DIR = "Dataset"
    CSV_PATH = os.path.join(DATA_DIR, "styles.csv")
    IMAGES_DIR = os.path.join(DATA_DIR, "train_images")
    BACKUP_DIR = os.path.join(DATA_DIR, "excluded_items")

    print("Preparing dataset...")
    results = prepare_dataset(DATA_DIR, CSV_PATH, IMAGES_DIR, BACKUP_DIR)
    valid_data, missing = connect_dataset(CSV_PATH, IMAGES_DIR)

    return valid_data, missing


if __name__ == "__main__":
    main()