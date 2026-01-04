import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import io

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Cafe Sales Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING (To match the Dark Dashboard look) ---
st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #31333f;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA CLEANING FUNCTION ---
@st.cache_data
def load_and_clean_data(file_path):
    # Handle Git Conflict Markers often found in 'dirty' files
    with open(file_path, 'r') as f:
        lines = f.readlines()
    filtered_lines = [line for line in lines if not line.startswith(('<<<<<<<', '=======', '>>>>>>>'))]
    df = pd.read_csv(io.StringIO("".join(filtered_lines)))
    
    # Clean Column Names
    df.columns = df.columns.str.strip()
    
    # Standardize 'Dirty' values
    df.replace(['ERROR', 'UNKNOWN'], np.nan, inplace=True)
    
    # Type Conversion
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')
    df['Total Spent'] = pd.to_numeric(df['Total Spent'], errors='coerce')
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
    
    # Imputation (Filling missing values)
    df['Quantity'] = df['Quantity'].fillna(df['Quantity'].median())
    df['Price Per Unit'] = df['Price Per Unit'].fillna(df['Price Per Unit'].median())
    df['Total Spent'] = df['Total Spent'].fillna(df['Total Spent'].median())
    df['Item'] = df['Item'].fillna(df['Item'].mode()[0])
    df['Location'] = df['Location'].fillna(df['Location'].mode()[0])
    
    return df

# Load Data
try:
    df_raw = load_and_clean_data('dirty_cafe_sales.csv')
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.title("☕ Cafe Sales")
st.sidebar.subheader("Navigate")
view = st.sidebar.radio("Choose view", ["Overview", "Visual Explorer", "Regression Model", "Data Table"])

st.sidebar.subheader("Filters")
selected_items = st.sidebar.multiselect("Item", options=df_raw['Item'].unique(), default=df_raw['Item'].unique())
selected_locs = st.sidebar.multiselect("Location", options=df_raw['Location'].unique(), default=df_raw['Location'].unique())

# Apply Filters
df = df_raw[(df_raw['Item'].isin(selected_items)) & (df_raw['Location'].isin(selected_locs))]

# --- HEADER & KPI CARDS ---
st.title("📊 Cafe Sales EDA Dashboard")
st.caption("Interactive analysis • regression model • cleaning pipeline")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Rows", f"{len(df)}")
kpi2.metric("Total Revenue", f"${df['Total Spent'].sum():,.0f}")
kpi3.metric("Avg Spent", f"${df['Total Spent'].mean():.2f}")
kpi4.metric("Median Price", f"${df['Price Per Unit'].median():.2f}")
kpi5.metric("Item Count", f"{df['Item'].nunique()}")

st.divider()

# --- MAIN VIEWS ---
if view == "Overview":
    col1, col2 = st.columns([2, 1])
    with col1:
        daily_sales = df.groupby(df['Transaction Date'].dt.date)['Total Spent'].sum().reset_index()
        fig = px.line(daily_sales, x='Transaction Date', y='Total Spent', title="Revenue over Time", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(df, names='Item', values='Total Spent', title="Revenue by Item", hole=0.4, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

elif view == "Visual Explorer":
    tab1, tab2 = st.tabs(["Distributions", "Relationships"])
    with tab1:
        fig = px.histogram(df, x="Total Spent", color="Item", marginal="box", title="Spending Distribution by Item", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = px.scatter(df, x="Quantity", y="Total Spent", color="Item", size="Price Per Unit", hover_data=['Location'], title="Quantity vs Total Spent", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

elif view == "Regression Model":
    st.subheader("Linear Regression: Predicting Total Spent")
    
    # ML Pre-processing
    ml_df = df.dropna(subset=['Total Spent', 'Quantity', 'Price Per Unit'])
    X = ml_df[['Quantity', 'Price Per Unit']]
    y = ml_df['Total Spent']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Metrics
    m1, m2 = st.columns(2)
    m1.metric("R² Score", f"{r2_score(y_test, predictions):.4f}")
    m2.metric("Mean Absolute Error", f"${mean_absolute_error(y_test, predictions):.2f}")
    
    # Plot
    res_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    # fig = px.scatter(res_df, x='Actual', y='Predicted', trendline="ols", title="Actual vs. Predicted", template="plotly_dark")
    fig = px.scatter(res_df, x='Actual', y='Predicted', title="Actual vs. Predicted", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif view == "Data Table":
    st.subheader("Raw Cleaned Data")
    st.dataframe(df, use_container_width=True)