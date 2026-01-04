import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def clean_cafe_data(df):
    """Professional cleaning with logic-based recovery."""
    df_cleaned = df.copy()
    
    # 1. Standardize invalid markers
    invalid_markers = ['ERROR', 'UNKNOWN', '', 'nan', 'none']
    df_cleaned.replace(invalid_markers, np.nan, inplace=True)
    
    # 2. Type Casting
    num_cols = ['Quantity', 'Price Per Unit', 'Total Spent']
    for col in num_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    df_cleaned['Transaction Date'] = pd.to_datetime(df_cleaned['Transaction Date'], errors='coerce')
    
    # 3. Intelligent Recovery (Recalculate Price * Qty if Total is ERROR)
    mask = df_cleaned['Total Spent'].isna() & df_cleaned['Quantity'].notna() & df_cleaned['Price Per Unit'].notna()
    df_cleaned.loc[mask, 'Total Spent'] = df_cleaned['Quantity'] * df_cleaned['Price Per Unit']
    
    # 4. Forward Fill missing categorical data
    for col in ['Item', 'Payment Method', 'Location']:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else "Other")
        
    return df_cleaned

def get_ml_forecast(df):
    """New Utility: Basic forecasting for the next period."""
    try:
        df_ml = df.dropna(subset=['Total Spent', 'Quantity', 'Price Per Unit'])
        X = df_ml[['Quantity', 'Price Per Unit']]
        y = df_ml['Total Spent']
        model = RandomForestRegressor(n_estimators=50)
        model.fit(X, y)
        return model
    except:
        return None