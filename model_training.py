from functions import analyze_dataset, prepare_dataset, connect_dataset, extract_combined_features, process_all_images
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

# Train and evaluate a random forest classifier
# Load and prepare data
df_feature_matrix = pd.read_csv('final_feature_matrix.csv')
X = df_feature_matrix.iloc[:, 1:-1]
y = df_feature_matrix.iloc[:, -1]

# Split and train the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_classifier.fit(X_train, y_train)

# Get predictions
y_pred = rf_classifier.predict(X_test)
y_pred_proba = rf_classifier.predict_proba(X_test)

# Print metrics first
print("Basic Classification Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Macro Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 1. Confusion Matrix Plot
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens')
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()  # Added to prevent label cutoff
plt.savefig('confusion_matrix.png')
plt.close()  # Close the figure to free memory

# 2. ROC Curves
plt.figure(figsize=(10, 8))
categories = np.unique(y)
colors = ['blue', 'red', 'green', 'purple']  # One color per category

for i, (color, category) in enumerate(zip(colors, categories)):
    y_test_binary = (y_test == category).astype(int)
    y_score = y_pred_proba[:, i]

    fpr, tpr, _ = roc_curve(y_test_binary, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{category} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves by Category')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.close()

print("\nPlots have been saved to:")
print("- confusion_matrix.png")
print("- roc_curves.png")

# Save the model
print("Saving the trained model...")
joblib.dump(rf_classifier, 'random_forest_model.joblib')
print("Model saved successfully!")