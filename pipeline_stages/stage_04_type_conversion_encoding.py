import pandas as pd
import numpy as np

def convert_types_and_encode(data):
    """Converts object columns to numeric if applicable, and one-hot encodes remaining categorical columns."""
    print(f"Data shape before type conversion and encoding: {data.shape}")

    # Identify object columns
    object_cols = data.select_dtypes(include=['object']).columns
    print(f"Columns initially detected as 'object' type: {list(object_cols)}")

    for col in object_cols:
        # Try to convert to numeric
        numeric_conversion = pd.to_numeric(data[col], errors='coerce')
        # Calculate successful conversion rate
        conversion_success_rate = 1 - (numeric_conversion.isna().sum() / len(data))
        
        if conversion_success_rate > 0.9:  # If over 90% converts successfully
            print(f"Column '{col}' appears to contain numeric data. Converting to numeric.")
            data[col] = numeric_conversion
        else:
            print(f"Column '{col}' will be treated as categorical.")

    # Re-identify categorical columns (those that are still object type)
    categorical_cols = data.select_dtypes(include=['object']).columns
    print(f"\nCategorical columns to be one-hot encoded: {list(categorical_cols)}")

    # One-hot encode the identified categorical columns
    if len(categorical_cols) > 0:
        data = pd.get_dummies(data, columns=list(categorical_cols), prefix=list(categorical_cols))
        print(f"Data shape after one-hot encoding: {data.shape}")
    else:
        print("No categorical columns found to one-hot encode.")

    print(f"Final data shape after type conversion and encoding: {data.shape}")
    
    # Save checkpoint
    output_path = 'pipeline_stages/04_type_converted_encoded_data.pkl'
    data.to_pickle(output_path)
    print(f"Type converted and encoded data saved to {output_path}")
    return data

if __name__ == '__main__':
    # For testing purposes, load the data from the previous stage
    input_data = pd.read_pickle('pipeline_stages/03_age_processed_data.pkl')
    processed_data = convert_types_and_encode(input_data)
    print("\nFirst 5 rows of processed data:")
    print(processed_data.head())
    print("\nInfo of processed data:")
    processed_data.info() 