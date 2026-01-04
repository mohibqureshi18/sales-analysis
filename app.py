import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import clean_cafe_data, get_ml_forecast

# Professional Configuration
st.set_page_config(page_title="Cafe Intelligence Pro", layout="wide", page_icon="☕")

# Custom CSS for a clean, modern UI
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .main { background-color: #f8f9fa; }
    </style>
""", unsafe_allow_html=True)

st.title("☕ Cafe Operations & Analytics Command Center")
st.sidebar.header("Data Hub")

uploaded_file = st.sidebar.file_uploader("Upload Daily Sales (CSV)", type="csv")

if uploaded_file:
    # Processing
    raw_df = pd.read_csv(uploaded_file)
    df = clean_cafe_data(raw_df)
    
    # Global Sidebar Filters
    locations = st.sidebar.multiselect("Filter by Location", options=df['Location'].unique(), default=df['Location'].unique())
    df_filtered = df[df['Location'].isin(locations)]

    # 1. Executive Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Gross Revenue", f"${df_filtered['Total Spent'].sum():,.2f}")
    with m2: st.metric("Avg Order Value", f"${df_filtered['Total Spent'].mean():,.2f}")
    with m3: st.metric("Volume", f"{int(df_filtered['Quantity'].sum())} items")
    with m4: st.metric("Active Locations", len(locations))

    # 2. THE INTERACTIVE WORKSPACE (Navigable Tabs)
    tab_sales, tab_geo, tab_health, tab_predict = st.tabs([
        "📈 Sales Performance", "📍 Location Intelligence", "🔍 Data Audit", "🔮 Forecast (ML)"
    ])

    with tab_sales:
        st.subheader("Revenue Trends & Distributions")
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Time Series with Plotly
            df_time = df_filtered.groupby('Transaction Date')['Total Spent'].sum().reset_index()
            fig_time = px.line(df_time, x='Transaction Date', y='Total Spent', title="Daily Revenue Stream")
            st.plotly_chart(fig_time, use_container_width=True)
            
        with col_right:
            # Item popularity
            fig_bar = px.bar(df_filtered.groupby('Item')['Quantity'].sum().reset_index(), 
                             x='Quantity', y='Item', orientation='h', title="Top Selling Items")
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_geo:
        st.subheader("Regional Breakdown")
        col_a, col_b = st.columns(2)
        with col_a:
            fig_pie = px.pie(df_filtered, names='Location', values='Total Spent', hole=0.5, title="Revenue by Channel")
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_b:
            fig_box = px.box(df_filtered, x='Location', y='Total Spent', color='Location', title="Ticket Size Variance")
            st.plotly_chart(fig_box, use_container_width=True)

    with tab_health:
        st.subheader("Data Cleaning Transparency")
        st.info(f"System automatically repaired {df['Total Spent'].isna().sum()} invalid entries.")
        st.dataframe(df_filtered, use_container_width=True)
        st.download_button("Export Enterprise Data", df_filtered.to_csv(index=False), "cleaned_sales.csv")

    with tab_predict:
        st.subheader("Random Forest Insights")
        st.write("Current model analyzing relationship between Quantity, Price, and Revenue.")
        model = get_ml_forecast(df)
        if model:
            q_input = st.slider("Simulate Quantity Sold", 1, 10, 5)
            p_input = st.number_input("Unit Price ($)", value=5.0)
            pred = model.predict([[q_input, p_input]])
            st.success(f"Predicted Transaction Value: **${pred[0]:.2f}**")
            
else:
    st.info("👋 Welcome! Please upload your sales data to the sidebar to populate the command center.")