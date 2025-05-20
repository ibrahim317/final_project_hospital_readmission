import pandas as pd
import numpy as np

def select_features_by_correlation(data, correlation_threshold=0.3):
    """Calculates correlation matrix and drops columns with low maximum correlation."""
    print(f"Data shape before correlation-based feature selection: {data.shape}")

    # Ensure all columns are numeric for correlation calculation
    # Attempt to convert non-numeric columns, coercing errors
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Drop columns that are entirely NaN after conversion (if any)
    data.dropna(axis=1, how='all', inplace=True)
    
    # Fill remaining NaNs with a value (e.g., 0 or median) before correlation, 
    # or handle them based on domain knowledge. Here, we fill with 0 for simplicity.
    # This might not be ideal for all datasets.
    data.fillna(0, inplace=True) 

    if data.empty or data.shape[1] == 0:
        print("Data is empty or has no columns after NaN handling. Skipping correlation analysis.")
        # Save checkpoint
        output_path = 'pipeline_stages/05_correlated_features_data.pkl'
        data.to_pickle(output_path)
        print(f"Empty or column-less data saved to {output_path}")
        return data
        
    corr_matrix = data.corr()

    low_corr_columns = []
    for col in corr_matrix.columns:
        # Get correlations excluding self (which is always 1.0)
        # Check if the column still exists in the matrix (it might have been dropped if all NaN)
        if col in corr_matrix:
            max_corr = corr_matrix[col].drop(col).abs().max()
            if pd.isna(max_corr) or max_corr < correlation_threshold: # handle NaN max_corr for columns with no variance or all NaNs
                low_corr_columns.append(col)

    if low_corr_columns:
        print(f"Found {len(low_corr_columns)} columns with max correlation < {correlation_threshold}: {low_corr_columns}")
        data.drop(columns=low_corr_columns, inplace=True, errors='ignore') # errors='ignore' in case a column was already removed
        print(f"Dataframe shape after dropping low correlation columns: {data.shape}")
    else:
        print(f"No columns found with max correlation < {correlation_threshold}.")

    print(f"Final data shape after correlation-based feature selection: {data.shape}")
    
    # Save checkpoint
    output_path = 'pipeline_stages/05_correlated_features_data.pkl'
    data.to_pickle(output_path)
    print(f"Correlated features data saved to {output_path}")
    return data

if __name__ == '__main__':
    # For testing purposes, load the data from the previous stage
    input_data = pd.read_pickle('pipeline_stages/04_type_converted_encoded_data.pkl')
    selected_data = select_features_by_correlation(input_data)
    print("\nFirst 5 rows of selected data:")
    print(selected_data.head())
    print("\nInfo of selected data:")
    selected_data.info() 