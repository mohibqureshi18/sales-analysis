import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def clean_cafe_data(df):
    """Advanced cleaning: Handles git markers, errors, and missing math."""
    # 1. Fix Headers if the CSV has Git conflict markers at the top
    if 'Transaction ID' not in df.columns:
        for i, row in df.iterrows():
            if 'Transaction ID' in [str(val) for val in row.values]:
                df.columns = row.values
                df = df.iloc[i+1:].reset_index(drop=True)
                break

    df_cleaned = df.copy()
    
    # 2. Standardize all forms of "Bad Data"
    bad_data = ['ERROR', 'UNKNOWN', 'nan', '<<<<<<< HEAD', '=======', '>>>>>>>']
    df_cleaned.replace(bad_data, np.nan, inplace=True)
    
    # 3. Convert types (Force numeric)
    num_cols = ['Quantity', 'Price Per Unit', 'Total Spent']
    for col in num_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    if 'Transaction Date' in df_cleaned.columns:
        df_cleaned['Transaction Date'] = pd.to_datetime(df_cleaned['Transaction Date'], errors='coerce')
    
    # 4. INTELLIGENT RECOVERY: If Total Spent is 'ERROR', calculate it manually
    if all(c in df_cleaned.columns for c in num_cols):
        missing_total = df_cleaned['Total Spent'].isna() & df_cleaned['Quantity'].notna() & df_cleaned['Price Per Unit'].notna()
        df_cleaned.loc[missing_total, 'Total Spent'] = df_cleaned['Quantity'] * df_cleaned['Price Per Unit']
    
    # 5. Fill Categorical gaps
    for col in ['Item', 'Payment Method', 'Location']:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].fillna("Other/Misc")
        
    return df_cleaned

def train_revenue_model(df):
    """Utility to build a predictive model based on your Jupyter Notebook logic."""
    try:
        data = df.dropna(subset=['Total Spent', 'Quantity', 'Price Per Unit'])
        if len(data) < 10: return None
        
        X = data[['Quantity', 'Price Per Unit']]
        y = data['Total Spent']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    except:
        return None