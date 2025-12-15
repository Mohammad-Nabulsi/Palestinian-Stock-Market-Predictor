import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def create_time_since_last(x):
    """
    Create a feature that calculates the time since the last trade for each stock.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'date' and 'stock' columns (if not specified).

    Returns:
    pd.DataFrame: DataFrame with an additional 'time_since_last' column.
    """
    # Create new DF from the passed one
    df = x.copy()
    
    df['date'] = df.index
    # Calculate time since last trade
    df['time_since_last'] = df['date'].diff().dt.days
    df.drop(columns='date', inplace=True)
    
    # Drop the first row of each stock
    #df_new = df_new.dropna(subset=['time_since_last'])
    
    return df

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

    
    drop_cols = ['yesterday', 'change_pctg']
    if 'tomorrow' in stock_new:
        drop_cols += ['tomorrow', 'change_from_tomorrow']

    stock_new.drop(columns=drop_cols, inplace=True)
    stock_new.rename(columns={'change_from_yesterday': 'change_pctg'}, inplace=True)

    return stock_new

def create_target_variable(df, classification=True,duration=1, direction=True, epsilon=0.001):
                                            
    """
    Create a target variable that indicates whether the stock price increased the next day.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'closing' and 'stock' columns (if not specified).
    classification (bool): If True, create a binary target variable. Default is True.
    duration (int): Number of days ahead to predict. Default is 1.
    direction (str): Direction of concern, possible values are True and False. Default is True and it indicates that we are concerned with spotting the increase. 0 inndicates that we are concerend in the decrease in the stock value.It's a classification only variable.

    Returns:
    pd.DataFrame: DataFrame with an additional '[clf/reg]_target_[duration]d_[direction]' column.
    """
    df_new = df.copy()
    
    
    df_new['tomorrow'] = df_new['closing'].shift(-1*duration)

    ret = (df_new['tomorrow'] - df_new['closing']) / df_new['closing']
    
    # Create target variable
    if classification:
        df_new[f'clf_target_{duration}d'] = 0
        if direction:
            df_new.loc[ret >  epsilon, f'clf_target_{duration}d'] =  1
        else:
            df_new.loc[ret <  epsilon, f'clf_target_{duration}d'] =  1
    else:
        df_new[f'reg_target_{duration}d'] = df_new['tomorrow'] - df_new['closing']
    
    
    df_new = df_new.dropna(subset=['tomorrow'])
    
    
    df_new.drop(columns=['tomorrow'], inplace=True)
    
    return df_new

def create_moving_averages(x, list=[20, 50, 200]):
    """
    x: must be an individual stock dataframe
    list: sorted ascendingly of wanted MAs
    Docstring for create_moving_averages
    
    :param df: Description
    :param list: Description
    """
    df = x.copy()

    for ma in list:
        df[f'{int(ma)}_MA'] = df['closing'].rolling(window=ma).mean()

    a = max(list)
    df['ma_ratio'] = df['closing'] / df[f'{int(a)}_MA']
    df['ma_fast_slow'] = df[f'{list[0]}_MA'] / df[f'{list[-1]}_MA']

    return df



def create_range(x):
    """
    x: must be an individual stock dataframe
    Docstring for create_moving_averages
    
    :param df: Description
    :param list: Description
    """
    df = x.copy()
    df['range'] = df['highest'] - df['lowest']

    return df

def create_daily_returns(x, list=[1, 3, 5]):
    """
    x: must be an individual stock dataframe
    Docstring for create_moving_averages
    
    :param df: Description
    :param list: Description
    """
    df = x.copy()
    lst = np.array(list)

    for day in lst:
        df[f'r{day}'] = df['closing'].pct_change(day)
    
    return df

def create_volatility(x, list=[10, 20]):
    """
    x: must be an individual stock dataframe
    Docstring for create_moving_averages
    
    :param df: Description
    :param list: Description
    """

    df = x.copy()

    for v in list:
        df[f'volatility_{v}d'] = df['r1'].rolling(v).std()
    
    return df

def create_rsi(x, list=[7, 14]):
    """
    x: must be an individual stock dataframe
    Docstring for create_moving_averages
    
    :param df: Description
    :param list: Description
    """

    df = x.copy()
    delta = df['closing'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    for window in list:
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        df[f'rsi_{window}'] = 100 - (100 / (1 + avg_gain / avg_loss))    

    return df


def create_liquidity(x):
    """
    x: must be an individual stock dataframe
    Docstring for create_moving_averages
    
    :param df: Description
    :param list: Description
    """

    df =x.copy()
    df['n_deals_change'] = df['n_deals'].pct_change()

    return df


def create_macd(x):
    df = x.copy()
    df['ema12'] = df['closing'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['closing'].ewm(span=26, adjust=False).mean()

    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    df.drop(columns=['ema12', 'ema26'], inplace=True)

    return df

def create_stochastic(x, window=14):
    df = x.copy()

    low14 = df['lowest'].rolling(window).min()
    high14 = df['highest'].rolling(window).max()

    df['stoch_k'] = 100 * ((df['closing'] - low14) / (high14 - low14))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['stoch_diff'] = df['stoch_k'] - df['stoch_d']
    df.drop(columns='stoch_k', inplace=True)
    return df