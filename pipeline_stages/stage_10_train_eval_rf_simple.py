import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

def train_evaluate_rf_simple(data):
    """Trains and evaluates a simpler Random Forest model."""
    print(f"Data shape for Simple RF model training: {data.shape}")

    # Define target and features
    target_col = 'readmitted_<30'
    if target_col not in data.columns:
        # Fallback if the primary target is missing - this should ideally not happen if data prep is consistent
        available_targets = [col for col in ['readmitted_>30', 'readmitted_NO'] if col in data.columns]
        if not available_targets:
            raise ValueError(f"Target column '{target_col}' and fallbacks not found in the data.")
        print(f"Warning: '{target_col}' not found. Using first available alternative: {available_targets[0]}")
        target_col = available_targets[0]
    
    Y = data[target_col]
    
    # Features to drop: IDs and all potential target-related columns
    cols_to_drop_ids = ['patient_nbr', 'encounter_id']
    cols_to_drop_targets = ['readmitted_<30', 'readmitted_>30', 'readmitted_NO']
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
    print(f"Target value counts:\n{Y.value_counts(normalize=True)}")

    # Train-test split (stratified if possible)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, 
                                                        stratify=Y if Y.nunique() > 1 else None, 
                                                        random_state=42)

    print("Training Simple Random Forest model...")
    rf = RandomForestClassifier(
        n_estimators=400,         # As in original
        class_weight='balanced',  # Handles imbalance
        max_depth=6,              # Prevents overfitting, as in original
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1] # Probability for the positive class

    print("\nSimple Random Forest Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    target_names = [str(c) for c in sorted(Y.unique())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    # Calculate AUC-ROC only if there are multiple classes in y_test
    if len(np.unique(y_test)) > 1:
        try:
            print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        except ValueError as e:
            print(f"Could not calculate AUC-ROC: {e}. Check if y_pred_proba is valid and y_test has multiple classes.")
    else:
        print("AUC-ROC not calculated because there is only one class in y_test.")

    # Feature importances
    try:
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        print("\nTop 10 Feature Importances:")
        print(importances.sort_values(ascending=False).head(10))
    except AttributeError:
        print("Could not retrieve feature importances.")

if __name__ == '__main__':
    # For testing purposes, load the data from the previous stage
    input_data = pd.read_pickle('pipeline_stages/06_outliers_handled_data.pkl')
    train_evaluate_rf_simple(input_data) 