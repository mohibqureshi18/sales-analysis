import streamlit as st
import pandas as pd
import plotly.express as px
from utils import clean_cafe_data, train_revenue_model

# 1. Page Configuration
st.set_page_config(page_title="Cafe Intelligence Suite", layout="wide", page_icon="☕")

# Professional UI Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px; padding: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

st.title("☕ Cafe Sales Intelligence Command Center")
st.sidebar.header("📁 Data Gateway")

# Logic to handle the file
uploaded_file = st.sidebar.file_uploader("Drop your Sales CSV here", type="csv")

if uploaded_file:
    # Processing
    raw_df = pd.read_csv(uploaded_file)
    df = clean_cafe_data(raw_df)
    
    # Global Sidebar Filters
    st.sidebar.subheader("Global Filters")
    selected_loc = st.sidebar.multiselect("Filter Locations", df['Location'].unique(), default=df['Location'].unique())
    filtered_df = df[df['Location'].isin(selected_loc)]

    # Top Row Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue", f"${filtered_df['Total Spent'].sum():,.2f}")
    m2.metric("Avg Order", f"${filtered_df['Total Spent'].mean():,.2f}")
    m3.metric("Volume", f"{int(filtered_df['Quantity'].sum())} Items")
    m4.metric("Transactions", len(filtered_df))

    # SINGLE SPACE NAVIGATION (Tabs)
    tab_growth, tab_geo, tab_predict, tab_audit = st.tabs([
        "📈 Sales & Growth", "📍 Regional Insights", "🔮 Prediction Engine", "🧹 Data Health Audit"
    ])

    with tab_growth:
        st.subheader("Performance Trends")
        col_ts, col_item = st.columns([2, 1])
        with col_ts:
            fig_line = px.line(filtered_df.groupby('Transaction Date')['Total Spent'].sum().reset_index(), 
                              x='Transaction Date', y='Total Spent', title="Daily Revenue Timeline")
            st.plotly_chart(fig_line, use_container_width=True)
        with col_item:
            fig_pie = px.pie(filtered_df, names='Item', values='Total Spent', title="Revenue Mix")
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab_geo:
        st.subheader("Location & Channel Analysis")
        c1, c2 = st.columns(2)
        with c1:
            fig_loc = px.bar(filtered_df.groupby('Location')['Total Spent'].sum().reset_index(), 
                             x='Location', y='Total Spent', color='Location', title="Revenue by Branch")
            st.plotly_chart(fig_loc, use_container_width=True)
        with c2:
            fig_pay = px.box(filtered_df, x='Payment Method', y='Total Spent', title="Ticket Size by Payment Type")
            st.plotly_chart(fig_pay, use_container_width=True)

    with tab_predict:
        st.subheader("Machine Learning: Revenue Predictor")
        st.write("This model uses Random Forest logic to predict order totals.")
        model = train_revenue_model(df)
        if model:
            ui_col1, ui_col2 = st.columns(2)
            with ui_col1:
                qty = st.slider("Simulate Quantity", 1, 20, 5)
                price = st.number_input("Unit Price ($)", value=3.5)
            with ui_col2:
                pred = model.predict([[qty, price]])
                st.metric("Estimated Transaction Value", f"${pred[0]:.2f}")
                st.info("The prediction is based on the historical patterns found in your CSV.")

    with tab_audit:
        st.subheader("Cleaning Transparency Report")
        st.write("Data automatically repaired by `utils.py`. You can now export the clean version.")
        st.dataframe(filtered_df, use_container_width=True)
        st.download_button("💾 Download Clean CSV", filtered_df.to_csv(index=False), "cafe_sales_cleaned.csv")

else:
    st.info("Please upload the 'dirty_cafe_sales.csv' from your project folder to begin.")