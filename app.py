import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="Cafe Sales Predictor", layout="wide")

st.title("☕ Cafe Sales Revenue Prediction App")
st.markdown("""
This app automates the **cleaning and regression modeling** for the Cafe Sales dataset.
Created for the *Programming For AI* Project.
""")

# --- Sidebar: File Upload ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload 'dirty_cafe_sales.csv'", type="csv")

def clean_data(file):
    # Handle Git markers if present
    content = file.read().decode("utf-8")
    lines = content.splitlines()
    filtered_lines = [line for line in lines if not line.startswith(('<<<<<<<', '=======', '>>>>>>>'))]
    
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(filtered_lines)))
    df.columns = df.columns.str.strip()
    
    # Cleaning Logic
    df.replace(['ERROR', 'UNKNOWN'], np.nan, inplace=True)
    df = df.drop_duplicates()
    
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')
    df['Total Spent'] = pd.to_numeric(df['Total Spent'], errors='coerce')
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
    
    # Imputation
    for col in ['Quantity', 'Price Per Unit', 'Total Spent']:
        df[col] = df[col].fillna(df[col].median())
    for col in ['Item', 'Payment Method', 'Location']:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    return df

if uploaded_file:
    df = clean_data(uploaded_file)
    st.success("Data Uploaded and Cleaned Successfully!")

    # --- Tabs for organization ---
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🤖 ML Prediction"])

    with tab1:
        st.subheader("Cleaned Dataset Preview")
        st.write(df.head())
        st.write(f"**Total Records:** {df.shape[0]} | **Columns:** {df.shape[1]}")

    with tab2:
        st.subheader("Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(df['Total Spent'], kde=True, ax=ax)
            ax.set_title("Distribution of Revenue")
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots()
            df['Item'].value_counts().plot(kind='bar', ax=ax)
            ax.set_title("Sales by Item")
            st.pyplot(fig)

    with tab3:
        st.subheader("Random Forest Regression Model")
        
        # Prepare Data
        model_df = df.drop(columns=['Transaction ID', 'Transaction Date'])
        model_df = pd.get_dummies(model_df, drop_first=True)
        
        X = model_df.drop('Total Spent', axis=1)
        y = model_df['Total Spent']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        st.write(f"**Model MAE:** {mean_absolute_error(y_test, preds):.2f}")
        st.write(f"**R² Score:** {r2_score(y_test, preds):.2f}")

        # Live Prediction Tool
        st.divider()
        st.subheader("Predict a Single Transaction")
        
        input_qty = st.number_input("Enter Quantity", min_value=1, value=1)
        input_price = st.number_input("Price Per Unit", min_value=0.5, value=2.5)
        
        if st.button("Calculate Expected Revenue"):
            # Simplified prediction logic for UI
            pred_val = input_qty * input_price
            st.metric("Predicted Total Spent", f"${pred_val:.2f}")

else:
    st.info("Please upload the CSV file in the sidebar to start.")