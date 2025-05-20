import sys
import os
# No longer need to modify sys.path if pipeline_stages is a package
# and main.py is run from the project root.

try:
    # Imports from the pipeline_stages package
    from pipeline_stages.stage_01_load_initial_clean import load_and_initial_clean_data
    from pipeline_stages.stage_02_feature_engineering import engineer_features
    from pipeline_stages.stage_03_process_age_feature import process_age
    from pipeline_stages.stage_04_type_conversion_encoding import convert_types_and_encode
    from pipeline_stages.stage_05_correlation_feature_selection import select_features_by_correlation
    from pipeline_stages.stage_06_outlier_handling import handle_outliers_iqr
    from pipeline_stages.stage_07_train_eval_rf_smote_gridsearch import train_evaluate_rf_smote_randomsearch
    from pipeline_stages.stage_08_train_eval_knn import train_evaluate_knn
    from pipeline_stages.stage_09_train_eval_svm import train_evaluate_svm
    from pipeline_stages.stage_10_train_eval_rf_simple import train_evaluate_rf_simple
    from pipeline_stages.stage_11_train_eval_logistic_regression import train_evaluate_logistic_regression
    from pipeline_stages.stage_12_train_eval_decision_tree import train_evaluate_decision_tree
except ImportError as e:
    print(f"Error importing from pipeline_stages package: {e}")
    if not os.path.isdir('pipeline_stages'):
        print("Error: 'pipeline_stages' directory not found.")
    else:
        print("Files in 'pipeline_stages':", os.listdir('pipeline_stages'))
    sys.exit(1)


def run_full_pipeline():
    """Runs the complete data processing and model training pipeline."""
    print("Starting Stage 1: Load and Initial Clean Data")
    data, IDs = load_and_initial_clean_data()
    print("Completed Stage 1.")
    print("-" * 50)

    print("Starting Stage 2: Feature Engineering")
    data = engineer_features(data)
    print("Completed Stage 2.")
    print("-" * 50)

    print("Starting Stage 3: Process Age Feature")
    data = process_age(data)
    print("Completed Stage 3.")
    print("-" * 50)

    print("Starting Stage 4: Type Conversion and Encoding")
    data = convert_types_and_encode(data)
    print("Completed Stage 4.")
    print("-" * 50)

    print("Starting Stage 5: Correlation-Based Feature Selection")
    data = select_features_by_correlation(data)
    print("Completed Stage 5.")
    print("-" * 50)

    print("Starting Stage 6: Outlier Handling")
    data = handle_outliers_iqr(data)
    print("Completed Stage 6. Preprocessing complete.")
    print("=" * 50)

    print("Starting Stage 7: Train and Evaluate Random Forest (SMOTE, RandomizedSearchCV)")
    train_evaluate_rf_smote_randomsearch(data)
    print("Completed Stage 7.")
    print("-" * 50)

    print("Starting Stage 8: Train and Evaluate KNN")
    train_evaluate_knn(data)
    print("Completed Stage 8.")
    print("-" * 50)

    # print("Starting Stage 9: Train and Evaluate SVM")
    # train_evaluate_svm(data)
    # print("Completed Stage 9.")
    # print("-" * 50)

    print("Starting Stage 10: Train and Evaluate Simple Random Forest")
    train_evaluate_rf_simple(data)
    print("Completed Stage 10.")
    print("-" * 50)

    print("Starting Stage 11: Train and Evaluate Logistic Regression")
    train_evaluate_logistic_regression(data)
    print("Completed Stage 11.")
    print("-" * 50)

    print("Starting Stage 12: Train and Evaluate Decision Tree")
    train_evaluate_decision_tree(data)
    print("Completed Stage 12.")
    print("=" * 50)

    print("Full pipeline execution completed.")

if __name__ == '__main__':
    if not os.path.isdir('pipeline_stages'):
        print("Critical Error: 'pipeline_stages' directory not found in the current working directory.")
        print(f"Current working directory: {os.getcwd()}")
        print("Please ensure 'pipeline_stages' is present and contains the stage scripts and __init__.py.")
        sys.exit(1)
        
    # The comments about renaming files are no longer needed as files were renamed in a previous step.
    # The imports are now structured to work with pipeline_stages as a package.
    run_full_pipeline()