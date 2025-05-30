import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler # Logistic Regression benefits from feature scaling

def train_evaluate_logistic_regression(data):
    """Scales features, trains and evaluates a Logistic Regression model."""
    print(f"Data shape for Logistic Regression model training: {data.shape}")

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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20, 
                                                        stratify=Y if Y.nunique() > 1 else None)
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Logistic Regression model...")
    # Add class_weight='balanced' and increase max_iter if it doesn't converge
    lgr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000, solver='liblinear')
    lgr.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = lgr.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nLogistic Regression Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    
    target_names = [str(c) for c in sorted(Y.unique())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

if __name__ == '__main__':
    # For testing purposes, load the data from the previous stage
    input_data = pd.read_pickle('pipeline_stages/06_outliers_handled_data.pkl')
    train_evaluate_logistic_regression(input_data) 