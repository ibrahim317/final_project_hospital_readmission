import pandas as pd
import numpy as np

def handle_outliers_iqr(data):
    """Removes rows with outliers based on the IQR method for numeric columns."""
    print(f"Data shape before outlier handling: {data.shape}")
    
    # Fill NaN values before outlier detection. Median is a common choice.
    # This is crucial as IQR calculations will fail or be skewed by NaNs.
    for col in data.select_dtypes(include=np.number).columns:
        if data[col].isnull().any():
            print(f"Filling NaNs in numeric column '{col}' with median before outlier detection.")
            data[col].fillna(data[col].median(), inplace=True)

    # Initialize a mask to keep all rows initially
    mask = pd.Series(True, index=data.index)

    numeric_cols = data.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        print("No numeric columns found to perform outlier handling.")
    else:
        print(f"Performing IQR outlier detection on columns: {list(numeric_cols)}")
        for columnName in numeric_cols:
            Q1 = data[columnName].quantile(0.25)
            Q3 = data[columnName].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Update the mask: keep rows where the current column's value is within bounds
            # Only consider non-NaN values for outlier detection bounds
            current_col_mask = data[columnName].between(lower_bound, upper_bound)
            mask &= current_col_mask
            
            num_outliers_col = (~current_col_mask).sum()
            if num_outliers_col > 0:
                print(f"Found {num_outliers_col} outliers in column '{columnName}'.")

    original_rows = len(data)
    data = data[mask]
    rows_removed = original_rows - len(data)
    
    print(f"\nRemoved {rows_removed} rows containing outliers.")
    print(f"Data shape after outlier handling: {data.shape}")
    
    # Save checkpoint
    output_path = 'pipeline_stages/06_outliers_handled_data.pkl'
    data.to_pickle(output_path)
    print(f"Outlier handled data saved to {output_path}")
    return data

if __name__ == '__main__':
    # For testing purposes, load the data from the previous stage
    input_data = pd.read_pickle('pipeline_stages/05_correlated_features_data.pkl')
    processed_data = handle_outliers_iqr(input_data)
    print("\nFirst 5 rows of processed data (after outlier handling):")
    print(processed_data.head())
    print("\nInfo of processed data:")
    processed_data.info() 