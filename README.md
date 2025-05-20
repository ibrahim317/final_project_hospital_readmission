# Hospital Readmission Prediction

This project aims to predict hospital readmission rates for diabetic patients using various machine learning models. The pipeline processes patient data through multiple stages of cleaning, feature engineering, and model training.

## Project Structure

```
Hospital-Readmission-Prediction/
├── data/                      # Data directory (not included in repo)
│   ├── diabetic_data.csv     # Main dataset
│   └── IDs_mapping.csv       # ID mappings
├── pipeline_stages/          # Pipeline processing stages
│   ├── stage_01_*.py        # Data loading and cleaning
│   ├── stage_02_*.py        # Feature engineering
│   └── ...                  # Additional stages
├── GUI/                     # GUI components
├── main.py                  # Main pipeline runner
├── full.py                  # Original full implementation
└── Instructions.txt         # Installation instructions
```

## Setup

1. Clone this repository
2. Follow the installation instructions in `Instructions.txt`
3. Place the required data files in the `data/` directory
4. Run the pipeline using `python main.py`

## Pipeline Stages

1. Load and Initial Clean Data
2. Feature Engineering
3. Process Age Feature
4. Type Conversion and Encoding
5. Correlation-Based Feature Selection
6. Outlier Handling
7. Train and Evaluate Random Forest (SMOTE, RandomizedSearchCV)
8. Train and Evaluate KNN
9. Train and Evaluate SVM
10. Train and Evaluate Simple Random Forest
11. Train and Evaluate Logistic Regression
12. Train and Evaluate Decision Tree

## Models

The project implements and compares several machine learning models:
- Random Forest (with SMOTE and RandomizedSearchCV)
- K-Nearest Neighbors
- Support Vector Machine
- Simple Random Forest
- Logistic Regression
- Decision Tree

Each model is evaluated using standard metrics including accuracy, precision, recall, and F1-score.