import pandas as pd
import numpy as np

def extract_midpoint(range_str):
    """Extracts midpoint from an age range string like '[0-10)'."""
    if pd.isna(range_str):
        return None
    try:
        # Ensure range_str is a string before stripping
        range_str = str(range_str).strip('[())')  # Remove brackets and potential trailing parenthesis
        # Handle cases like '70-80' or '[70-80)'
        if '-' in range_str:
            lower, upper = map(int, range_str.split('-'))
            return (lower + upper) / 2
        return None # Or handle single numbers if they exist
    except ValueError:
        print(f"Could not convert range: {range_str}")
        return None

def process_age(data):
    """Converts age ranges to midpoints and drops the original age column."""
    print(f"Data shape before age processing: {data.shape}")

    if 'age' not in data.columns:
        print("'age' column not found. Skipping age processing.")
    else:
        print("Original 'age' column (first few values):")
        print(data['age'].head())
        
        data['age_numeric'] = data['age'].apply(extract_midpoint)
        
        print("\nConverted 'age_numeric' column (first few values):")
        print(data['age_numeric'].head())
        
        null_count = data['age_numeric'].isna().sum()
        print(f"\nNumber of 'age_numeric' values that couldn't be converted or were already null: {null_count}")
        
        if null_count > 0:
            print("\nSample of problematic 'age' values (if any that weren't already NaN):")
            problematic = data[data['age_numeric'].isnull() & data['age'].notnull()]['age'].unique()
            print(problematic[:5])
            
        data.drop('age', axis=1, inplace=True)
        print("Dropped original 'age' column.")

    print(f"Data shape after age processing: {data.shape}")
    
    # Save checkpoint
    output_path = 'pipeline_stages/03_age_processed_data.pkl'
    data.to_pickle(output_path)
    print(f"Age processed data saved to {output_path}")
    return data

if __name__ == '__main__':
    # For testing purposes, load the data from the previous stage
    input_data = pd.read_pickle('pipeline_stages/02_engineered_features_data.pkl')
    age_processed_data = process_age(input_data)
    print("\nFirst 5 rows of age processed data:")
    print(age_processed_data.head())
    if 'age_numeric' in age_processed_data.columns:
        print("\n'age_numeric' column info:")
        print(age_processed_data['age_numeric'].describe())
    print("\nInfo of age processed data:")
    age_processed_data.info() 