import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def analyze_feature_importance(rf_classifier, feature_names, top_n=15):
    # Get feature importances
    importances = rf_classifier.feature_importances_

    # Create DataFrame of features and their importance scores
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # Plot top N features
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature',
                data=feature_importance.head(top_n))
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    return feature_importance

if __name__ == "__main__":
    # Load the trained model
    rf_classifier = joblib.load('random_forest_model_V3.joblib')

    # Load feature matrix to get feature names
    df_feature_matrix = pd.read_csv('final_feature_matrix.csv')
    feature_names = df_feature_matrix.iloc[:, 1:-1].columns  # All columns except ID and target

    # Analyze feature importance
    importance_df = analyze_feature_importance(rf_classifier, feature_names)

    # Print top features and their importance scores
    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15))