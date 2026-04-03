import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Telco Customer Churn Prediction", page_icon="📡", layout="wide")

# Custom CSS for Dark Premium Theme
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    /* Main Headers */
    h1, h2, h3, h4 {
        color: #58a6ff !important;
        font-weight: 700;
    }
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #238636;
        color: white;
        border: None;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: 0.3s background-color ease;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        color: white;
    }
    /* Cards for metrics/info */
    div[data-testid="stMetricValue"] {
        color: #58a6ff;
        font-size: 2.5rem;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* General Inputs */
    .stSelectbox>div>div>div, .stNumberInput>div>div>div {
        background-color: #21262d;
        color: white;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rf_churn_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoders.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, scaler, encoders

def plot_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {'suffix': "%", 'font': {'color': '#c9d1d9'}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability", 'font': {'size': 24, 'color': '#58a6ff'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#c9d1d9"},
            'bar': {'color': "#238636" if probability < 0.5 else "#da3633"},
            'bgcolor': "#21262d",
            'borderwidth': 2,
            'bordercolor': "#30363d",
            'steps': [
                {'range': [0, 50], 'color': '#161b22'},
                {'range': [50, 100], 'color': '#161b22'}],
            'threshold': {
                'line': {'color': "#da3633", 'width': 4},
                'thickness': 0.75,
                'value': 50}}))
    
    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font={'color': "#c9d1d9"})
    return fig

def main():
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>📡 Telco Customer Churn Prediction</h1>", unsafe_allow_html=True)
    
    # Load data for EDA
    try:
        df = load_raw_data()
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return
        
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        st.markdown("Select an analytical view:")
        page = st.radio("Select an analytical view:", ["📊 Explanatory Data Analysis", "🔮 Predict Churn"], label_visibility="collapsed")
        st.markdown("---")
        st.info("The ML pipeline uses a Random Forest Classifier trained on 7000+ customer records.")
    
    if page == "📊 Explanatory Data Analysis":
        st.header("Exploratory Data Analysis")
        st.markdown("<p style='color: #8b949e;'>Analyze historical user data to find patterns associated with churn.</p>", unsafe_allow_html=True)
        
        # Display key metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Customers", f"{len(df):,}")
        m2.metric("Overall Churn Rate", f"{(df['Churn'] == 'Yes').mean() * 100:.1f}%")
        
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        m3.metric("Avg Monthly Revenue", f"${df['MonthlyCharges'].mean():.2f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Churn Overview")
            churn_counts = df['Churn'].value_counts().reset_index()
            churn_counts.columns = ['Churn Status', 'Count']
            fig_churn = px.pie(churn_counts, names='Churn Status', values='Count', hole=0.4, 
                               color='Churn Status', color_discrete_map={'No': '#2ea043', 'Yes': '#da3633'})
            fig_churn.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#c9d1d9")
            st.plotly_chart(fig_churn, use_container_width=True)
            
        with col2:
            st.subheader("Tenure Distribution")
            fig_tenure = px.histogram(df, x="tenure", color="Churn", nbins=30, opacity=0.8,
                                     barmode="overlay", color_discrete_map={'No': '#58a6ff', 'Yes': '#da3633'})
            fig_tenure.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#c9d1d9")
            st.plotly_chart(fig_tenure, use_container_width=True)
            
        st.subheader("Monthly Charges Impact")
        fig_charges = px.box(df, x="Churn", y="MonthlyCharges", color="Churn", 
                            color_discrete_map={'No': '#58a6ff', 'Yes': '#da3633'})
        fig_charges.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#c9d1d9")
        st.plotly_chart(fig_charges, use_container_width=True)
        
        st.subheader("Raw Dataset Explorer")
        st.dataframe(df.head(100), use_container_width=True)
        
    elif page == "🔮 Predict Churn":
        st.header("Predict Customer Churn")
        st.markdown("<p style='color: #8b949e;'>Enter a customer's specific profile to calculate their likelihood of churning.</p>", unsafe_allow_html=True)
        
        try:
            model, scaler, encoders = load_artifacts()
        except Exception as e:
            st.error(f"Could not load ML artifacts. Have you run the training scripts? Error: {e}")
            return
            
        st.markdown("### Client Profile Form")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Demographics**")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
                
            with col2:
                st.markdown("**Services Overview**")
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                device_protect = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                
            with col3:
                st.markdown("**Additional Services & Billing**")
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                
            col_a, col_b = st.columns(2)
            with col_a:
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
            with col_b:
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
                
            submit_button = st.form_submit_button(label="Generate Prediction 🚀")
            
        if submit_button:
            input_dict = {
                'gender': gender,
                'SeniorCitizen': senior,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protect,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless,
                'PaymentMethod': payment,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            input_df = pd.DataFrame([input_dict])
            
            for col, le in encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = le.transform(input_df[col])
                    except Exception as e:
                        pass
                        
            num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            input_df[num_cols] = scaler.transform(input_df[num_cols])
            
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                st.plotly_chart(plot_gauge(probability), use_container_width=True)
            
            with res_col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if prediction == 1:
                    st.error(f"⚠️ **High Churn Risk Detected**\n\nThe model predicts an **{probability:.1%}** chance that this customer will cancel their service. Consider proactive retention offers.")
                else:
                    st.success(f"✅ **Low Churn Risk**\n\nThe model predicts an **{probability:.1%}** chance of churn. This customer is likely stable.")

if __name__ == "__main__":
    main()
