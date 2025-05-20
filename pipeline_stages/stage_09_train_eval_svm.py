import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

def train_evaluate_svm(data):
    """Scales features, trains and evaluates an SVM model."""
    print(f"Data shape for SVM model training: {data.shape}")

    # Define target and features
    if 'readmitted_<30' in data.columns:
        Y_col = 'readmitted_<30'
        # Features to drop: IDs and other target-related columns if they exist
        cols_to_drop_explicit = ['patient_nbr', 'encounter_id', 'readmitted_>30', 'readmitted_NO']
        cols_to_drop = [col for col in cols_to_drop_explicit if col in data.columns] + [Y_col]
        X = data.drop(columns=cols_to_drop, errors='ignore')
        Y = data[Y_col]
    else:
        print("Target column 'readmitted_<30' not found. Using the last column as target and rest as features.")
        if data.shape[1] < 2:
             raise ValueError("Data must have at least two columns to define X and Y.")
        X = data.iloc[:, :-1]
        Y = data.iloc[:, -1]
        print(f"Using target: {data.columns[-1]}")

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

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=30, stratify=Y if Y.nunique() > 1 else None)

    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Initialize and train SVM classifier (linear kernel as in original)
    # For large datasets, consider LinearSVC or SVC with adjusted parameters (e.g., C, gamma for non-linear)
    print("Training SVM model (linear kernel)...")
    # Add class_weight='balanced' if classes are imbalanced
    clf = svm.SVC(kernel='linear', class_weight='balanced', random_state=42) 
    clf.fit(x_train_scaled, y_train)

    # Cross-validation scores on training data
    print("Calculating cross-validation scores...")
    try:
        # Ensure there are enough samples per class for CV if stratifying (default for SVC)
        scores = cross_val_score(clf, x_train_scaled, y_train, cv=min(5, y_train.value_counts().min()), scoring='accuracy') # Adjust cv folds if classes are small
        print(f"Cross-validation accuracy scores: {scores}")
        print(f"Mean CV accuracy: {scores.mean():.4f}")
    except ValueError as e:
        print(f"Could not perform cross-validation, possibly due to class imbalance or small sample size: {e}")
        print("Skipping CV.")

    # Evaluate on the test set
    y_predict = clf.predict(x_test_scaled)
    test_accuracy = accuracy_score(y_test, y_predict)

    print("\nSVM Model Evaluation on Test Set:")
    print(f"Test set accuracy: {test_accuracy:.4f}")
    
    target_names = [str(c) for c in sorted(Y.unique())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_predict, target_names=target_names, zero_division=0))

    # Print some example predictions
    print("\nSample Predictions (first 10 from test set):")
    for i in range(min(10, len(y_test))):
        print(f"True: {y_test.iloc[i]}, Predicted: {y_predict[i]}")

if __name__ == '__main__':
    # For testing purposes, load the data from the previous stage
    input_data = pd.read_pickle('pipeline_stages/06_outliers_handled_data.pkl')
    train_evaluate_svm(input_data) 