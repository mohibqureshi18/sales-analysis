import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import io

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Professional Cafe Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- THEME & CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetric"] {
        background-color: #fff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #31333f;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA CLEANING ---
@st.cache_data
def load_and_clean_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    filtered_lines = [line for line in lines if not line.startswith(('<<<<<<<', '=======', '>>>>>>>'))]
    df = pd.read_csv(io.StringIO("".join(filtered_lines)))
    
    df.columns = df.columns.str.strip()
    df.replace(['ERROR', 'UNKNOWN'], np.nan, inplace=True)
    
    for col in ['Quantity', 'Price Per Unit', 'Total Spent']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
        
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
    df['Item'] = df['Item'].fillna(df['Item'].mode()[0])
    df['Location'] = df['Location'].fillna(df['Location'].mode()[0])
    df['Payment Method'] = df['Payment Method'].fillna(df['Payment Method'].mode()[0])
    
    return df

df_raw = load_and_clean_data('dirty_cafe_sales.csv')

# --- SIDEBAR ---
st.sidebar.title("☕ Cafe Analytics")
view = st.sidebar.radio("Navigate", ["Overview", "Visual Explorer", "Correlation", "Regression Model"])

# Filters
st.sidebar.markdown("---")
items = st.sidebar.multiselect("Select Items", options=df_raw['Item'].unique(), default=df_raw['Item'].unique())
locs = st.sidebar.multiselect("Select Locations", options=df_raw['Location'].unique(), default=df_raw['Location'].unique())

df = df_raw[(df_raw['Item'].isin(items)) & (df_raw['Location'].isin(locs))]

# --- DASHBOARD HEADER ---
st.title("📊 Cafe Sales Dashboard")
st.caption("Enhanced Visualizations & Regression Analysis")

# KPI Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions", len(df))
m2.metric("Total Revenue", f"${df['Total Spent'].sum():,.0f}")
m3.metric("Avg Sale", f"${df['Total Spent'].mean():.2f}")
m4.metric("Top Item", df['Item'].mode()[0])

st.divider()

# --- VIEWS ---
if view == "Overview":
    col1, col2 = st.columns(2)
    
    with col1:
        # NEW: Bar Chart for Revenue by Item
        item_rev = df.groupby('Item')['Total Spent'].sum().sort_values(ascending=False).reset_index()
        fig1 = px.bar(item_rev, x='Item', y='Total Spent', color='Item', 
                      title="Total Revenue by Item Type", template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        # Time Trend
        daily = df.groupby(df['Transaction Date'].dt.date)['Total Spent'].sum().reset_index()
        fig2 = px.line(daily, x='Transaction Date', y='Total Spent', 
                       title="Daily Revenue Trend", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    # NEW: Summary Table
    st.subheader("Item Performance Summary")
    summary = df.groupby('Item').agg({
        'Quantity': 'sum',
        'Total Spent': 'sum',
        'Price Per Unit': 'mean'
    }).rename(columns={'Price Per Unit': 'Avg Price'}).reset_index()
    st.dataframe(summary.style.format({"Total Spent": "${:,.2f}", "Avg Price": "${:,.2f}"}), use_container_width=True)

elif view == "Visual Explorer":
    tab1, tab2, tab3 = st.tabs(["Location Analysis", "Payment Methods", "Distributions"])
    
    with tab1:
        # NEW: Sunburst Chart (Hierarchical view)
        fig_sun = px.sunburst(df, path=['Location', 'Item'], values='Total Spent',
                              title="Revenue Breakdown: Location > Item", template="plotly_dark")
        st.plotly_chart(fig_sun, use_container_width=True)
        
    with tab2:
        # NEW: Horizontal Bar Chart for Payment Methods
        pay_df = df['Payment Method'].value_counts().reset_index()
        fig_pay = px.bar(pay_df, x='count', y='Payment Method', orientation='h',
                         title="Popularity of Payment Methods", template="plotly_dark")
        st.plotly_chart(fig_pay, use_container_width=True)
        
    with tab3:
        fig_box = px.box(df, x="Item", y="Total Spent", color="Location", 
                         title="Spending Range by Item and Location", template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)

elif view == "Correlation":
    st.subheader("Feature Correlation Matrix")
    # NEW: Heatmap showing relationships between numbers
    corr = df[['Quantity', 'Price Per Unit', 'Total Spent']].corr()
    fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', 
                         title="Numerical Correlation", template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)

elif view == "Regression Model":
    st.header("📈 Sales Prediction")
    X = df[['Quantity', 'Price Per Unit']]
    y = df['Total Spent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    c1, c2 = st.columns(2)
    c1.metric("R² Score (Accuracy)", f"{r2_score(y_test, y_pred):.4f}")
    c2.metric("Mean Absolute Error", f"${mean_absolute_error(y_test, y_pred):.2f}")
    
    res_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    fig_reg = px.scatter(res_df, x='Actual', y='Predicted', trendline="ols", 
                         title="Actual vs Predicted Sales", template="plotly_dark")
    st.plotly_chart(fig_reg, use_container_width=True)