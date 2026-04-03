# Telco Customer Churn Prediction Project

This project provides a full end-to-end Machine Learning pipeline to predict customer churn for a telecommunications company. It includes data preprocessing, exploratory data analysis (EDA), model training, and a Streamlit interactive dashboard.

## Project Structure
- `data/`: Contains the raw dataset and the processed dataset (after running the preprocessing script).
- `src/`: Python scripts for automated data preprocessing and model training.
- `models/`: Saved Machine Learning models (Random Forest), standard scalers, and label encoders.
- `dashboard/`: A Streamlit web application for interactive data visualization and live churn prediction.
- `requirements.txt`: Project dependencies.

## Dashboard Screenshots
![Dashboard View 1](images/Screenshot%202026-04-03%20153048.png)
![Dashboard View 2](images/Screenshot%202026-04-03%20153116.png)
![Dashboard View 3](images/Screenshot%202026-04-03%20153138.png)
![Dashboard View 4](images/Screenshot%202026-04-03%20153155.png)
![Dashboard View 5](images/Screenshot%202026-04-03%20153430.png)

## Key Findings & Model Evaluation
The dataset was cleaned and encoded, and a **Random Forest Classifier** was trained on the processed data.
- **Model Accuracy**: ~80%
- Features such as `tenure`, `MonthlyCharges`, and internal service attributes were key in the model's predictions.

## Installation & Setup

1. **Download Dataset**
   The raw dataset is excluded from version control to save space. Download the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the `data/` directory.

2. **Install Dependencies**
   It's recommended to use a virtual environment. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preprocessing**
   Run the data preprocessing script to clean the data and save label encoders:
   ```bash
   python src/data_preprocessing.py
   ```

3. **Train the Model**
   Run the model training script to train the Random Forest model and save the scaler:
   ```bash
   python src/model_training.py
   ```

## Running the Dashboard

You can start the interactive Streamlit dashboard by running:
```bash
streamlit run dashboard/app.py
```

The dashboard includes two tabs:
- **EDA**: Visualizations for churn distributions, tenure distributions, and the raw dataset.
- **Predict Churn**: A prediction form allowing you to input new customer details and receive an immediate churn probability score directly from the trained model.
