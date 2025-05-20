import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def train_evaluate_rf_smote_randomsearch(data):
    """Applies SMOTE, trains and evaluates a Random Forest model using RandomizedSearchCV for hyperparameter tuning."""
    print(f"Data shape for RF (SMOTE, RandomizedSearchCV) model training: {data.shape}")

    # Define target and features
    target_cols = ['readmitted_<30', 'readmitted_>30', 'readmitted_NO']
    # Check which target columns exist in the dataframe
    existing_target_cols = [col for col in target_cols if col in data.columns]
    
    if not existing_target_cols:
        raise ValueError("None of the target columns ('readmitted_<30', 'readmitted_>30', 'readmitted_NO') found in the data.")
    
    # Using 'readmitted_<30' as the primary target for this model, if available
    Y_col = 'readmitted_<30'
    if Y_col not in data.columns:
        print(f"Warning: '{Y_col}' not found, using the first available target: {existing_target_cols[0]}")
        Y_col = existing_target_cols[0]
    
    Y = data[Y_col]
    
    # Features to drop: IDs and other target-related columns
    cols_to_drop = ['patient_nbr', 'encounter_id'] + existing_target_cols
    # Ensure only existing columns are dropped
    cols_to_drop_existing = [col for col in cols_to_drop if col in data.columns]
    
    X = data.drop(columns=cols_to_drop_existing, errors='ignore') # errors='ignore' is belt-and-suspenders

    # Fill any remaining NaNs in features with median (as SMOTE cannot handle NaNs)
    for col in X.columns:
        if X[col].isnull().any():
            print(f"Filling NaNs in feature column '{col}' with median before SMOTE.")
            X[col].fillna(X[col].median(), inplace=True)

    if X.empty:
        raise ValueError("Feature set X is empty after dropping columns and handling NaNs.")
    if Y.empty:
        raise ValueError("Target set Y is empty.")

    print(f"Shape of X: {X.shape}, Shape of Y: {Y.shape}")
    print(f"Target value counts:\n{Y.value_counts(normalize=True)}")

    # Split the dataset (use stratify if highly imbalanced, though SMOTE handles imbalance)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # Apply SMOTE to training data
    print("Applying SMOTE to training data...")
    sm = SMOTE(random_state=42)
    try:
        x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    except ValueError as e:
        print(f"Error during SMOTE: {e}. This might be due to insufficient samples of a class.")
        print("Skipping SMOTE and proceeding with original training data for this model.")
        x_train_res, y_train_res = x_train, y_train

    print(f"Shape of x_train_res: {x_train_res.shape}, Shape of y_train_res: {y_train_res.shape}")
    print(f"Resampled target value counts:\n{pd.Series(y_train_res).value_counts(normalize=True)}")

    # Define hyperparameter grid
    param_dist = {
        'n_estimators': [100, 200, 300], # Reduced from original to speed up for example
        'max_depth': [None, 5, 7, 9], # Original had range(9), using specific values
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    # Initialize and search
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                     n_iter=10, # Reduced from 13 for speed
                                     cv=3, scoring='recall', # Recall is often good for imbalanced medical data
                                     n_jobs=-1, random_state=42, verbose=1)

    print("Starting RandomizedSearchCV...")
    random_search.fit(x_train_res, y_train_res)

    print(f'Best parameters found: {random_search.best_params_}')

    # Evaluate
    best_rf = random_search.best_estimator_
    y_pred = best_rf.predict(x_test)

    print(f'Accuracy = {accuracy_score(y_test, y_pred) * 100:.2f}%')
    print("Classification Report:")
    # Ensure target_names match the actual classes present in y_test and y_pred
    # If Y is boolean or 0/1, class names might be simple.
    # For this example, assuming binary classification with classes 0 and 1 (No/Yes)
    target_names = ['No', 'Yes'] if len(np.unique(y_test)) == 2 else [str(c) for c in sorted(np.unique(y_test))]
    if len(target_names) != len(np.unique(np.concatenate((y_test, y_pred)))) and len(np.unique(y_test)) == 2:
         print("Adjusting target names for classification report based on actual unique values.")
         # This handles if y_pred only contains one class, but y_test has two.
         # Or if classes are not exactly 0 and 1 but, say, True/False.
         unique_labels = sorted(pd.unique(np.concatenate((y_test, y_pred))))
         target_names = [f'Class {i}' for i in unique_labels]
    
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Feature importances
    try:
        importances = pd.Series(best_rf.feature_importances_, index=X.columns)
        print("\nTop 10 Feature Importances:")
        print(importances.sort_values(ascending=False).head(10))
    except AttributeError:
        print("Could not retrieve feature importances (possibly due to an issue with the best_rf model).")

    # Tree visualization (optional, can be time-consuming for large forests/trees)
    try:
        plt.figure(figsize=(20, 10))
        plot_tree(best_rf.estimators_[0], feature_names=X.columns, class_names=target_names, filled=True, max_depth=3, rounded=True)
        plt.title("One Tree from the Best Random Forest (SMOTE, RandomizedSearchCV)")
        plt.savefig('pipeline_stages/07_rf_smote_randomsearch_tree.png')
        plt.show()
        print("Saved sample tree visualization to pipeline_stages/07_rf_smote_randomsearch_tree.png")
    except Exception as e:
        print(f"Could not plot tree: {e}")

if __name__ == '__main__':
    # For testing purposes, load the data from the previous stage
    input_data = pd.read_pickle('pipeline_stages/06_outliers_handled_data.pkl')
    train_evaluate_rf_smote_randomsearch(input_data) 