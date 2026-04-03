import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def load_processed_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data file not found at: {file_path}. Please run data_preprocessing.py first.")
    return pd.read_csv(file_path)

def train_model(X_train, y_train):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
    model.fit(X_train, y_train)
    return model

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed_churn_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    print("Loading processed data...")
    df = load_processed_data(data_path)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Scale numerical features (tenure, MonthlyCharges, TotalCharges)
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest model...")
    model = train_model(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Save the model and scaler
    model_path = os.path.join(models_dir, 'rf_churn_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    main()
