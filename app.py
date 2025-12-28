import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sales Dashboard", layout="wide")

st.title("Sales Analysis & Cleaning Tool")
st.write("Upload your 'dirty' sales CSV to automatically clean data and visualize insights.")

# 1. Dynamic File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
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

    # Feature Engineering for Filters
    df_cleaned['Month'] = df_cleaned['Transaction Date'].dt.month_name()
    df_cleaned['Weekday'] = df_cleaned['Transaction Date'].dt.day_name()

    # 3. Sidebar Filters
    st.sidebar.header("Filter Options")
    selected_month = st.sidebar.multiselect("Select Month", options=df_cleaned['Month'].unique(), default=df_cleaned['Month'].unique())
    selected_day = st.sidebar.multiselect("Select Day of Week", options=df_cleaned['Weekday'].unique(), default=df_cleaned['Weekday'].unique())

    # Apply Filters
    filtered_df = df_cleaned[(df_cleaned['Month'].isin(selected_month)) & (df_cleaned['Weekday'].isin(selected_day))]

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

    # Row 2: Bar Charts
    st.write("### Mean Sales by Category")
    cat_col = st.selectbox("View Mean Sales by:", ['Item', 'Payment Method', 'Location'])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    order_data = filtered_df.groupby(cat_col)['Total Spent'].mean().sort_values(ascending=False)
    sns.barplot(x=order_data.index, y=order_data.values, palette='viridis', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Data Preview
    st.write("### Cleaned Data Preview (Filtered)")
    st.dataframe(filtered_df.head(50))

else:

    st.info("Please upload a CSV file to begin.")
