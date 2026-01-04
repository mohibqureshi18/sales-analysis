import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="Cafe Sales Dashboard", layout="wide")

# ==========================================
# 1. DATA RESCUE FUNCTION
# ==========================================
def load_and_fix_data(filepath):
    """Removes Git conflict markers and loads the CSV safely."""
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Filter out Git conflict markers (<<<<, ====, >>>>)
        clean_lines = [
            line for line in lines 
            if not (line.startswith('<<<<<<<') or 
                    line.startswith('=======') or 
                    line.startswith('>>>>>>>'))
        ]
        
        # Write to a temporary file to load into Pandas
        temp_file = "temp_clean_data.csv"
        with open(temp_file, 'w') as f:
            f.writelines(clean_lines)
            
        df = pd.read_csv(temp_file)
        df.columns = df.columns.str.strip() # Remove whitespace from headers
        return df
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None

# ==========================================
# 2. DATA PROCESSING
# ==========================================
raw_df = load_and_fix_data('dirty_cafe_sales.csv')

if raw_df is not None:
    df = raw_df.copy()
    
    # Replace placeholder error strings with NaN
    invalid_entries = ['ERROR', 'UNKNOWN', '']
    df.replace(invalid_entries, np.nan, inplace=True)
    
    # Robust type conversion (errors='coerce' turns "HEAD" or "<<<" into NaN)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')
    df['Total Spent'] = pd.to_numeric(df['Total Spent'], errors='coerce')
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')

    # Drop rows where critical data for visualization is missing
    df = df.dropna(subset=['Total Spent', 'Item'])

    # ==========================================
    # 3. STREAMLIT UI
    # ==========================================
    st.title("☕ Cafe Sales Performance Dashboard")
    st.markdown("---")

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Options")
    
    # Location Filter
    all_locations = sorted(df['Location'].dropna().unique().tolist())
    selected_locations = st.sidebar.multiselect(
        "Select Locations", 
        options=all_locations, 
        default=all_locations
    )
    
    # Filtered Dataframe
    filtered_df = df[df['Location'].isin(selected_locations)]

    # --- Key Metrics Row ---
    col1, col2, col3 = st.columns(3)
    
    total_rev = filtered_df['Total Spent'].sum()
    avg_spend = filtered_df['Total Spent'].mean()
    total_qty = filtered_df['Quantity'].sum()

    col1.metric("Total Revenue", f"${total_rev:,.2f}")
    col2.metric("Average Transaction", f"${avg_spend:,.2f}")
    col3.metric("Total Items Sold", f"{int(total_qty) if not np.isnan(total_qty) else 0:,}")

    st.markdown("---")

    # --- Visualization Row 1 ---
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Revenue by Item Category")
        item_data = filtered_df.groupby('Item')['Total Spent'].sum().reset_index()
        fig_bar = px.bar(
            item_data.sort_values('Total Spent', ascending=False), 
            x='Item', 
            y='Total Spent',
            color='Total Spent',
            color_continuous_scale='Viridis',
            labels={'Total Spent': 'Total Revenue ($)'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with row1_col2:
        st.subheader("Preferred Payment Methods")
        pay_data = filtered_df['Payment Method'].value_counts().reset_index()
        fig_pie = px.pie(
            pay_data, 
            names='Payment Method', 
            values='count',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Visualization Row 2 ---
    st.subheader("Sales Trend Over Time")
    if not filtered_df['Transaction Date'].isna().all():
        trend_data = filtered_df.groupby('Transaction Date')['Total Spent'].sum().reset_index()
        fig_line = px.line(
            trend_data, 
            x='Transaction Date', 
            y='Total Spent',
            markers=True,
            labels={'Total Spent': 'Daily Revenue ($)'}
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No valid transaction dates available to show trends.")

    # --- Data Preview (Expander) ---
    with st.expander("View Cleaned Data Sample"):
        st.dataframe(filtered_df.head(50), use_container_width=True)

else:
    st.error("🚨 **File Not Found:** Please ensure 'dirty_cafe_sales.csv' is in the same directory as this script.")