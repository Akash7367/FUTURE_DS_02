# Telco Customer Churn Prediction Project

This project provides a full end-to-end Machine Learning pipeline to predict customer churn for a telecommunications company. It includes data preprocessing, exploratory data analysis (EDA), model training, and a Streamlit interactive dashboard.

## Project Structure
- `data/`: Contains the raw dataset and the processed dataset (after running the preprocessing script).
- `src/`: Python scripts for automated data preprocessing and model training.
- `models/`: Saved Machine Learning models (Random Forest), standard scalers, and label encoders.
- `dashboard/`: A Streamlit web application for interactive data visualization and live churn prediction.
- `requirements.txt`: Project dependencies.

## Dashboard Screenshots
### 1. 🌟 **Dashboard Overview & Key Metrics**
> **_A clean, Glassmorphism-themed dark mode interface showing total customers, churn rates, and average revenue metrics._**

![Dashboard View 1](images/Screenshot%202026-04-03%20153048.png)

### 2. 📊 **Interactive Data Visualizations**
> **_Interactive Plotly charts detailing Churn Distributions and Customer Tenure, allowing for detailed hover insights._**

![Dashboard View 2](images/Screenshot%202026-04-03%20153116.png)

### 3. 🔍 **Feature Impact Analysis & Raw Data**
> **_Exploration of how Monthly Charges influence churn probability, alongside a view of the preprocessed raw dataset._**

![Dashboard View 3](images/Screenshot%202026-04-03%20153138.png)

### 4. 🔮 **Live Prediction Client Form**
> **_A comprehensive input form allowing users to simulate different customer profiles including demographics, services, and billing._**

![Dashboard View 4](images/Screenshot%202026-04-03%20153155.png)

### 5. 🎯 **Instant AI Churn Risk Gauge**
> **_Real-time AI probability calculation converted into an interactive Gauge chart, cleanly indicating whether the customer is High or Low risk._**

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
