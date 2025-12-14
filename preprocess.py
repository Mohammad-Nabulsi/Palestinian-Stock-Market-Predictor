import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def datize_date(df, date_col='date'):
    df_new = df.copy()
    df_new['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    return df_new

def numerize_value_and_volume(df, value_exist=True):
    df_new = df.copy()
    if value_exist:
        df_new['value'] = df_new['value'].str.replace(',', '').astype(float)
    df_new['volume'] = df_new['volume'].str.replace(',', '').astype(float)
    return df_new

def remove_nas(x):

    df = x.copy()
    print(df.isna().sum())
    df = df[199:].copy()
    df['rsi_7'] = df['rsi_7'].fillna(50)
    df['rsi_14'] = df['rsi_14'].fillna(50)
    print("="*30)
    print("Remaining Nulls", df.isna().sum().sum())

    return df

def time_series_split(x, ratio=0.2):
    df = x.copy()
    all_cols = df.columns

    target_col = df.columns[2]
    feature_cols = [c for c in all_cols if c != target_col]

    X = df[feature_cols]
    y = df[target_col].values

    n = len(df)
    test_size = max( int(ratio * n), 100 )  
    split_idx = n - test_size

    X_trainval, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_trainval, y_test = y[:split_idx], y[split_idx:]
    
    return X_trainval, X_test, y_trainval, y_test, feature_cols, target_col

def preprocessor(x, feature_cols, target_col):

    df = x.copy()
    cat_cols = ['day_of_week']                 
    bin_cols = ['first_week_of_month']           
    num_cols = [c for c in feature_cols if c not in cat_cols + bin_cols]

    sk = df[num_cols].skew()
    mean_cols = [c for c in num_cols if abs(sk[c]) < 0.5]
    median_cols = [c for c in num_cols if abs(sk[c]) >= 0.5]

    median_pre = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
    
    mean_pre = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),    
    ('scaler', StandardScaler())
])
    
    binary_pre = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])
    
    categorical_pre = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
    
    preprocess = ColumnTransformer(transformers=[
    ('mean', mean_pre, mean_cols),
    ('median', median_pre, median_cols),
    ('bin', binary_pre, bin_cols),
    ('cat', categorical_pre, cat_cols)
], remainder='drop')
    
    return preprocess