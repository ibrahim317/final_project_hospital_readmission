import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
from dotenv import load_dotenv, dotenv_values

def load_and_initial_clean_data():
    """Loads data and performs initial cleaning as per full.py"""
    # loading variables from .env file
    load_dotenv()
    
    data = pd.read_csv("data/diabetic_data.csv")
    IDs = pd.read_csv("data/IDs_mapping.csv")
    
    # Initial data cleaning
    for (columnName, columnData) in data.items():
        print('Column Name : ', columnName)
        data.replace('?', None, inplace=True)
        print(data.isnull().sum())
    
    # Drop columns with high missing values
    for (columnName, columnData) in data.items():
        if(data[columnName].isnull().sum()/data[columnName].count() >= 0.9):
            data.drop(columnName, axis=1, inplace=True)
    
    # Save the processed dataframe
    output_path = 'pipeline_stages/01_cleaned_data.pkl'
    data.to_pickle(output_path)
    print(f"Cleaned data saved to {output_path}")
    
    return data, IDs

if __name__ == '__main__':
    data, IDs = load_and_initial_clean_data()
    print("\nFirst 5 rows of cleaned data:")
    print(data.head())
    print("\nInfo of cleaned data:")
    data.info() 