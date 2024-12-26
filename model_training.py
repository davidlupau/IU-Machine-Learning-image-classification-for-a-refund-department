import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def main():
    try:
        # Load and prepare data
        print("Loading feature matrix...")
        feature_matrix = pd.read_csv('final_feature_matrix.csv')

        print(f"\nInitial feature matrix shape: {feature_matrix.shape}")
        print(f"Categories present: {feature_matrix['category'].unique()}")

        # Clean the data
        print("\nCleaning feature matrix...")
        # Get all numeric columns except 'image_id'
        feature_columns = feature_matrix.select_dtypes(include=[np.number]).columns
        X = feature_matrix[feature_columns]
        y = feature_matrix['category']

        # Handle infinite and very large values
        print("Handling infinite and extreme values...")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean(axis=0))  # Fill NaN with column means

        # Calculate quantiles for each column
        lower_bounds = X.quantile(0.01, axis=0)
        upper_bounds = X.quantile(0.99, axis=0)

        # Clip values column by column
        for column in X.columns:
            X[column] = X[column].clip(lower=lower_bounds[column], upper=upper_bounds[column])

        print(f"Features shape after cleaning: {X.shape}")
        print(f"Number of categories: {len(y.unique())}")
        print("Checking for infinite values:", np.any(np.isinf(X)))
        print("Checking for NaN values:", np.any(np.isnan(X)))

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
            verbose=2
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
        plt.figure(figsize=(15, 12))  # Increased size for many categories
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens')
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

        # ROC Curves
        plt.figure(figsize=(10, 8))
        categories = np.unique(y)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))

        for i, (color, category) in enumerate(zip(colors, categories)):
            y_test_binary = (y_test == category).astype(int)
            y_score = y_pred_proba[:, i]
            fpr, tpr, _ = roc_curve(y_test_binary, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'{category[:15]}... (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves by Category')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('roc_curves.png')
        plt.close()

        print("\nPlots have been saved to:")
        print("- confusion_matrix.png")
        print("- roc_curves.png")

        # Save the model
        print("\nSaving the model...")
        joblib.dump(rf_classifier, 'random_forest_model.joblib')
        print("Model saved successfully!")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Current working directory:", os.getcwd())
        print("Files in directory:", os.listdir())


if __name__ == "__main__":
    main()