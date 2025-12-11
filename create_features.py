import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def create_time_since_last_trade(df, stock_col='stock', date_col='date'):
    """
    Create a feature that calculates the time since the last trade for each stock.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'date' and 'stock' columns (if not specified).

    Returns:
    pd.DataFrame: DataFrame with an additional 'time_since_last' column.
    """
    # Create new DF from the passed one
    df_new = df.copy()

    # Sort by stock_id and timestamp
    df_new = df_new.sort_values(by=[stock_col, date_col])
    
    # Calculate time since last trade
    df_new['time_since_last'] = df_new.groupby(stock_col)[date_col].diff().dt.days
    
    # Drop the first row of each stock
    df_new = df_new.dropna(subset=['time_since_last'])
    
    return df_new

def fix_stock_change_pctg_leakge(stock, stock_name):
    stock_new = stock.copy()
    
    stock_new['yesterday'] = stock_new['closing'].shift(1)
    stock_new['change_from_yesterday'] = (
        (stock_new['closing'] - stock_new['yesterday']) / stock_new['yesterday'] * 100
    )

    if stock_name == 'AIB':
        stock_new['tomorrow'] = stock_new['closing'].shift(-1)
        stock_new['change_from_tomorrow'] = (
            (stock_new['tomorrow'] - stock_new['closing']) / stock_new['closing'] * 100
        )
        print(stock_new[['change_pctg', 'change_from_yesterday', 'change_from_tomorrow']].head())

    # Clean + rename
    drop_cols = ['yesterday', 'change_pctg']
    if 'tomorrow' in stock_new:
        drop_cols += ['tomorrow', 'change_from_tomorrow']

    stock_new.drop(columns=drop_cols, inplace=True)
    stock_new.rename(columns={'change_from_yesterday': 'change_pctg'}, inplace=True)

    return stock_new

def create_target_variable(df, classification=True, Regression=False, 
                                            duration=1, closing_col='closing',
                                            stock_col='stock', direction='inc'):
    """
    Create a target variable that indicates whether the stock price increased the next day.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'closing' and 'stock' columns (if not specified).
    classification (bool): If True, create a binary target variable. Default is True.
    Regression (bool): If True, create a regression target variable. Default is False.
    duration (int): Number of days ahead to predict. Default is 1.
    direction (str): Direction of concern, possible values are 'inc' or 'dec' or 'N'. Default is 'inc' and it indicates that we are concerned with spotting the increase/decrease and no change for the value (N) in stock prices. It's a classification only variable.

    Returns:
    pd.DataFrame: DataFrame with an additional '[clf/reg]_target_[duration]d_[direction]' column.
    """
    df_new = df.copy()
    
    # Calculate next day's closing price
    df_new['tomorrow'] = df_new.groupby(stock_col)[closing_col].shift(-1*duration)
    
    # Create target variable
    if classification:
        if direction == 'inc':
            df_new[f'clf_target_{duration}d_{direction}'] = (df_new['tomorrow'] > df_new[closing_col]).astype(int)
        elif direction == 'dec':
            df_new[f'clf_target_{duration}d_{direction}'] = (df_new['tomorrow'] < df_new[closing_col]).astype(int)
        elif direction == '-':
            df_new[f'clf_target_{duration}d_{direction}'] = (df_new['tomorrow'] == df_new[closing_col]).astype(int)
    
    elif Regression:
        df_new[f'reg_target_{duration}d'] = df_new['tomorrow'] - df_new[closing_col]
    
    # Drop rows where next day's closing price is NaN
    df_new = df_new.dropna(subset=['tomorrow'])
    
    # Drop the helper column
    df_new.drop(columns=['tomorrow'], inplace=True)
    
    return df_new






