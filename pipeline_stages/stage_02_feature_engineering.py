import pandas as pd
import numpy as np

def engineer_features(data):
    """Maps IDs to descriptions and performs one-hot encoding."""
    print(f"Data shape before feature engineering: {data.shape}")

    admission_type_map = {
        1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
        5: "Not Available", 6: "NULL", 7: "Trauma Center", 8: "Not Mapped"
    }

    discharge_disposition_map = {
        1: "Discharged to home",
        2: "Discharged/transferred to another short term hospital",
        3: "Discharged/transferred to SNF",
        4: "Discharged/transferred to ICF",
        5: "Discharged/transferred to another type of inpatient care institution",
        6: "Discharged/transferred to home with home health service",
        7: "Left AMA",
        8: "Discharged/transferred to home under care of Home IV provider",
        9: "Admitted as an inpatient to this hospital",
        10: "Neonate discharged to another hospital for neonatal aftercare",
        11: "Expired",
        12: "Still patient or expected to return for outpatient services",
        13: "Hospice / home",
        14: "Hospice / medical facility",
        15: "Discharged/transferred within this institution to Medicare approved swing bed",
        16: "Discharged/transferred/referred another institution for outpatient services",
        17: "Discharged/transferred/referred to this institution for outpatient services",
        18: "NULL",
        19: "Expired at home. Medicaid only, hospice.",
        20: "Expired in a medical facility. Medicaid only, hospice.",
        21: "Expired, place unknown. Medicaid only, hospice.",
        22: "Discharged/transferred to another rehab facility including rehab units of a hospital",
        23: "Discharged/transferred to a long term care hospital",
        24: "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare",
        25: "Not Mapped",
        26: "Unknown/Invalid",
        27: "Discharged/transferred to a federal health care facility",
        28: "Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital",
        29: "Discharged/transferred to a Critical Access Hospital (CAH)",
        30: "Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere"
    }

    admission_source_map = {
        1: "Physician Referral",
        2: "Clinic Referral",
        3: "HMO Referral",
        4: "Transfer from a hospital",
        5: "Transfer from a Skilled Nursing Facility (SNF)",
        6: "Transfer from another health care facility",
        7: "Emergency Room",
        8: "Court/Law Enforcement",
        9: "Not Available",
        10: "Transfer from critical access hospital",
        11: "Normal Delivery",
        12: "Premature Delivery",
        13: "Sick Baby",
        14: "Extramural Birth",
        15: "Not Available",
        17: "NULL",
        18: "Transfer From Another Home Health Agency",
        19: "Readmission to Same Home Health Agency",
        20: "Not Mapped",
        21: "Unknown/Invalid",
        22: "Transfer from hospital inpt/same fac reslt in a sep claim",
        23: "Born inside this hospital",
        24: "Born outside this hospital",
        25: "Transfer from Ambulatory Surgery Center",
        26: "Transfer from Hospice"
    }

    data['admission_type_desc'] = data['admission_type_id'].map(admission_type_map)
    data['discharge_desc'] = data['discharge_disposition_id'].map(discharge_disposition_map)
    data['admission_source_desc'] = data['admission_source_id'].map(admission_source_map)
    
    # One-hot encode the new descriptive columns
    data = pd.get_dummies(data, columns=['admission_type_desc'], prefix='adm_type')
    data = pd.get_dummies(data, columns=['discharge_desc'], prefix='dis_desc')
    data = pd.get_dummies(data, columns=['admission_source_desc'], prefix='adm_src')
    
    # Drop the original ID columns
    if 'admission_type_id' in data.columns:
        data.drop('admission_type_id', axis=1, inplace=True)
    if 'discharge_disposition_id' in data.columns:
        data.drop('discharge_disposition_id', axis=1, inplace=True)
    if 'admission_source_id' in data.columns:
        data.drop('admission_source_id', axis=1, inplace=True)
        
    print(f"Data shape after feature engineering: {data.shape}")
    
    # Save checkpoint
    output_path = 'pipeline_stages/02_engineered_features_data.pkl'
    data.to_pickle(output_path)
    print(f"Engineered features data saved to {output_path}")
    return data

if __name__ == '__main__':
    # For testing purposes, load the data from the previous stage
    input_data = pd.read_pickle('pipeline_stages/01_cleaned_data.pkl')
    engineered_data = engineer_features(input_data)
    print("\nFirst 5 rows of engineered data:")
    print(engineered_data.head())
    print("\nInfo of engineered data:")
    engineered_data.info() 