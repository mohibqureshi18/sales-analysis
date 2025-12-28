import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sales Dashboard", layout="wide")

st.title("Sales Analysis & Cleaning Tool")
st.info("Currently displaying data from: **dirty_cafe_sales.csv**")

# 1. Direct File Loading (Replaces st.file_uploader)
try:
    df = pd.read_csv("dirty_cafe_sales.csv")
    df_cleaned = df.copy()

    # 2. Automated Cleaning Logic
    invalid_values = ['ERROR', 'UNKNOWN', '']
    df_cleaned.replace(invalid_values, np.nan, inplace=True)

    numerical_cols = ['Quantity', 'Price Per Unit', 'Total Spent']
    for col in numerical_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    df_cleaned['Transaction Date'] = pd.to_datetime(df_cleaned['Transaction Date'], errors='coerce')
    
    # Fill missing dates/numbers for filtering
    mode_date = df_cleaned['Transaction Date'].mode()[0]
    df_cleaned['Transaction Date'] = df_cleaned['Transaction Date'].fillna(mode_date)
    
    for col in numerical_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # 3. Sidebar Filters
    st.sidebar.header("Filter Options")
    items = st.sidebar.multiselect("Select Items", options=df_cleaned['Item'].unique().tolist(), default=df_cleaned['Item'].unique().tolist())
    
    filtered_df = df_cleaned[df_cleaned['Item'].isin(items)]

    # 4. Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${filtered_df['Total Spent'].sum():,.2f}")
    col2.metric("Avg Transaction", f"${filtered_df['Total Spent'].mean():,.2f}")
    col3.metric("Total Items Sold", int(filtered_df['Quantity'].sum()))

    # 5. Visualizations
    st.subheader("Visual Analysis")
    
    # Row 1: Heatmap and Distribution
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(filtered_df[numerical_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    with row1_col2:
        st.write("### Sales Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['Total Spent'], kde=True, color='skyblue', ax=ax)
        st.pyplot(fig)

except FileNotFoundError:
    st.error("Error: 'dirty_cafe_sales.csv' not found. Please ensure the file is in the same directory as this script.")