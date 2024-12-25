import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def main():
    # Load preprocessed data
    print("Loading feature matrix...")
    feature_matrix = pd.read_csv('final_feature_matrix.csv')

    # Clean and prepare data
    print("Preparing data for training...")
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
    feature_matrix = feature_matrix.fillna(0)
    X = feature_matrix.iloc[:, 1:-1]
    y = feature_matrix.iloc[:, -1]

    # Split and train
    print("\nSplitting data and training model...")
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define parameter grid for search
    print("\nDefining parameter grid for GridSearchCV...")
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    }

    # Create base model
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Create GridSearchCV object
    print("\nStarting GridSearchCV...")
    grid_search = GridSearchCV(
    rf_base,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2  # Add verbosity to see progress
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Print results
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print("\nBest cross-validation score:")
    print(f"{grid_search.best_score_:.4f}")

    # Use the best model for predictions
    rf_classifier = grid_search.best_estimator_
    y_pred = rf_classifier.predict(X_test)
    y_pred_proba = rf_classifier.predict_proba(X_test)

    # Print metrics
    print("\nTest Set Performance Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Macro Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens')
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # ROC Curves
    plt.figure(figsize=(10, 8))
    categories = np.unique(y)
    colors = ['blue', 'red', 'green', 'purple']

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
        print("\nSaving the enhanced model...")
        joblib.dump(rf_classifier, 'random_forest_model.joblib')
        print("Model saved successfully!")
    else:
        print("Feature matrix creation failed")


if __name__ == "__main__":
    main()