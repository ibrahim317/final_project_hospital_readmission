import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_evaluate_knn(data):
    """Scales features, trains and evaluates a KNN model using GridSearchCV for hyperparameter tuning."""
    print(f"Data shape for KNN model training: {data.shape}")

    # Define target and features
    # Attempt to use 'readmitted_<30' as target, or the last column if not present.
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
    
    # Fill NaNs in target with mode (common for classification tasks if minor)
    if Y.isnull().any():
        print(f"Filling NaNs in target column '{Y.name}' with mode.")
        Y.fillna(Y.mode()[0], inplace=True) # Fill with the first mode if multiple

    if X.empty or Y.empty:
        raise ValueError("Feature set X or target set Y is empty after pre-processing.")

    print(f"Shape of X: {X.shape}, Shape of Y: {Y.shape}")
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=30, stratify=Y if Y.nunique() > 1 else None)

    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Define hyperparameter grid for KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15], # Original had 11, 13, 15
        'weights': ['uniform', 'distance'], # Original had just distance
        'metric': ['euclidean', 'manhattan'] # Original had just euclidean
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    
    print("Starting GridSearchCV for KNN...")
    grid_search.fit(x_train_scaled, y_train)

    best_knn = grid_search.best_estimator_
    y_predict = best_knn.predict(x_test_scaled)

    print("\nKNN Model Evaluation:")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score (accuracy): {grid_search.best_score_:.4f}")
    print(f"Test set accuracy: {accuracy_score(y_test, y_predict):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_predict))
    
    target_names = [str(c) for c in sorted(Y.unique())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_predict, target_names=target_names, zero_division=0))

if __name__ == '__main__':
    # For testing purposes, load the data from the previous stage
    input_data = pd.read_pickle('pipeline_stages/06_outliers_handled_data.pkl')
    train_evaluate_knn(input_data) 