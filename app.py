import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SETTINGS & STYLE ---
st.set_page_config(page_title="Cafe Sales Analysis", layout="wide")
sns.set_theme(style="whitegrid")

st.title("☕ Cafe Sales Analysis Dashboard ☕")
st.markdown("This dashboard analyzes a fixed dataset of cafe transactions to identify sales patterns and correlations.")

# --- 2. REQUIREMENT: FIXED DATA LOAD ---
# We load the file directly from your repository
@st.cache_data # This keeps the app fast by only loading data once
def load_and_clean_data():
    df = pd.read_csv('dirty_cafe_sales.csv')
    
    # --- AUTOMATED CLEANING ENGINE ---
    df_cleaned = df.copy()
    df_cleaned.replace(['ERROR', 'UNKNOWN', ''], np.nan, inplace=True)
    
    for col in ['Quantity', 'Price Per Unit', 'Total Spent']:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    df_cleaned['Transaction Date'] = pd.to_datetime(df_cleaned['Transaction Date'], errors='coerce')
    df_cleaned['Transaction Date'] = df_cleaned['Transaction Date'].fillna(df_cleaned['Transaction Date'].mode()[0])
    
    # Feature Engineering for Filters
    df_cleaned['Month'] = df_cleaned['Transaction Date'].dt.month_name()
    df_cleaned['Weekday'] = df_cleaned['Transaction Date'].dt.day_name()
    
    return df_cleaned

df_cleaned = load_and_clean_data()

# --- 3. REQUIREMENT: DATE FILTERS IN SIDEBAR ---
st.sidebar.header("Filter Visuals")
selected_months = st.sidebar.multiselect(
    "Select Month(s)", 
    options=df_cleaned['Month'].unique(), 
    default=df_cleaned['Month'].unique()
)
selected_days = st.sidebar.multiselect(
    "Select Weekday(s)", 
    options=df_cleaned['Weekday'].unique(), 
    default=df_cleaned['Weekday'].unique()
)

# Apply Filters
filtered_df = df_cleaned[
    (df_cleaned['Month'].isin(selected_months)) & 
    (df_cleaned['Weekday'].isin(selected_days))
]

# --- 4. REQUIREMENT: THE VISUALS ---
if not filtered_df.empty:
    # Top Level Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Transactions", len(filtered_df))
    m2.metric("Total Revenue", f"${filtered_df['Total Spent'].sum():,.2f}")
    m3.metric("Avg. Item Price", f"${filtered_df['Price Per Unit'].mean():.2f}")

    st.divider()

    # Visuals Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Correlation: Price vs Quantity")
        fig1, ax1 = plt.subplots()
        sns.heatmap(filtered_df[['Quantity', 'Price Per Unit', 'Total Spent']].corr(), annot=True, cmap='YlGnBu', ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("Spending Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(filtered_df['Total Spent'], bins=15, kde=True, color='brown', ax=ax2)
        st.pyplot(fig2)

    st.divider()

    # Visuals Row 2
    st.subheader("Mean Sales by Item")
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    filtered_df.groupby('Item')['Total Spent'].mean().sort_values().plot(kind='bar', color='peru', ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

else:

    st.error("Please select at least one Month and one Weekday in the sidebar to view charts.")
