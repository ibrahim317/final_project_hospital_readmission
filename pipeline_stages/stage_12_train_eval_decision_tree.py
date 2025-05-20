import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def train_evaluate_decision_tree(data):
    """Trains and evaluates a Decision Tree model, and plots feature importances."""
    print(f"Data shape for Decision Tree model training: {data.shape}")

    # Define target and features
    target_col = 'readmitted_<30'
    if target_col not in data.columns:
        available_targets = [col for col in ['readmitted_>30', 'readmitted_NO'] if col in data.columns]
        if not available_targets:
            raise ValueError(f"Target column '{target_col}' and fallbacks not found in the data.")
        print(f"Warning: '{target_col}' not found. Using first available alternative: {available_targets[0]}")
        target_col = available_targets[0]
    
    Y = data[target_col]
    
    cols_to_drop_ids = ['patient_nbr', 'encounter_id']
    cols_to_drop_targets = ['readmitted_<30', 'readmitted_>30', 'readmitted_NO']
    # Ensure we only try to drop columns that exist, and don't drop the current target_col from X
    cols_to_drop = cols_to_drop_ids + [tc for tc in cols_to_drop_targets if tc in data.columns and tc != target_col]
    
    X = data.drop(columns=[col for col in cols_to_drop if col in data.columns] + [target_col])

    # Fill any remaining NaNs in features with median
    for col in X.columns:
        if X[col].isnull().any():
            print(f"Filling NaNs in feature column '{col}' with median.")
            X[col].fillna(X[col].median(), inplace=True)
            
    # Fill NaNs in target with mode
    if Y.isnull().any():
        print(f"Filling NaNs in target column '{Y.name}' with mode.")
        Y.fillna(Y.mode()[0], inplace=True)

    if X.empty or Y.empty:
        raise ValueError("Feature set X or target set Y is empty after pre-processing.")

    print(f"Shape of X: {X.shape}, Shape of Y: {Y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42,
                                                        stratify=Y if Y.nunique() > 1 else None)

    print("Training Decision Tree model...")
    # Using parameters from the original script
    clf = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=9, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nDecision Tree Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    
    target_names = [str(c) for c in sorted(Y.unique())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Feature importances
    try:
        importances = clf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        print("\nTop 10 Feature Importances:")
        print(feature_importance_df.head(10))

        # Plotting top 15 feature importances
        plt.figure(figsize=(12, 8))
        top_n = 15
        # Ensure we don't try to plot more features than available
        num_features_to_plot = min(top_n, len(feature_importance_df))
        plt.barh(feature_importance_df['Feature'][:num_features_to_plot][::-1], 
                 feature_importance_df['Importance'][:num_features_to_plot][::-1])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Top {num_features_to_plot} Feature Importances in Decision Tree")
        plt.tight_layout()
        plt.savefig('pipeline_stages/12_decision_tree_feature_importances.png')
        plt.show()
        print("Saved feature importance plot to pipeline_stages/12_decision_tree_feature_importances.png")

    except Exception as e:
        print(f"Could not plot feature importances: {e}")

if __name__ == '__main__':
    # For testing purposes, load the data from the previous stage
    input_data = pd.read_pickle('pipeline_stages/06_outliers_handled_data.pkl')
    train_evaluate_decision_tree(input_data) 