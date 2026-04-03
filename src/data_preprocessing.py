import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    """Load the dataset from the specified file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """Clean the data (handle missing values, convert types)."""
    # TotalCharges is object, need to convert to numeric. 
    # Some rows have ' ' (empty string) for TotalCharges, which causes errors.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with 0 (since they belong to customers with 0 tenure)
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Drop customerID as it's not useful for prediction
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    return df

def preprocess_data(df):
    """Encode categorical variables and scale numerical variables."""
    # Define categorical and numerical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Target variable 'Churn' should be 0 and 1
    if 'Churn' in categorical_columns:
        categorical_columns.remove('Churn')
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode categorical features
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    return df, label_encoders

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    processed_data_path = os.path.join(base_dir, 'data', 'processed_churn_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    print("Loading data...")
    df = load_data(data_path)
    
    print("Cleaning data...")
    df = clean_data(df)
    
    print("Preprocessing data (Encoding)...")
    df, encoders = preprocess_data(df)
    
    print(f"Saving processed data to {processed_data_path}")
    df.to_csv(processed_data_path, index=False)
    
    import joblib
    encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
    joblib.dump(encoders, encoders_path)
    print(f"Encoders saved to {encoders_path}")
    
    print("Data processing complete!")

if __name__ == "__main__":
    main()
