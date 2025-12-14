import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distributions(df,  target_col=None):
    """
    Plot the distributions of specified features. If a target column is provided,
    plot the distributions conditioned on the target variable.

    Parameters:
    df (pd.DataFrame): DataFrame containing the features and target.
    feature_cols (list): List of feature column names to plot.
    target_col (str, optional): Target column name for conditional distributions.

    Returns:
    None
    """
    feature_cols = list(df.columns)
    feature_cols.remove("clf_target_1d")
    num_features = len(feature_cols)
    fig, axes = plt.subplots(num_features, 1, figsize=(8, 5 * num_features))

    if num_features == 1:
        axes = [axes]

    for ax, feature in zip(axes, feature_cols):
        if target_col:
            sns.kdeplot(data=df, x=feature, hue=target_col, ax=ax, fill=True)
            ax.set_title(f'Distribution of {feature} by {target_col}')
        else:
            sns.kdeplot(data=df, x=feature, ax=ax, fill=True)
            ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')

    plt.tight_layout()
    plt.show()


def plot_initial(x, stock_name):
    df = x.copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    axes[0].plot(df.index, df['closing'], label='Closing Price', color='blue')
    axes[0].plot(df.index, df['opening'], label='Opening Price', color='red', alpha=0.4)
    axes[0].legend()
    axes[0].set_xlabel("Date", fontsize=14)
    axes[0].set_ylabel("Price", fontsize=14)
    axes[0].set_title(f"{stock_name} CLosing vs Opening Price", fontsize=20)

    axes[1].plot(df.index, df['highest'], label='Highest Price', color='blue')
    axes[1].plot(df.index, df['lowest'], label='Lowest Price', color='red', alpha=0.4)
    axes[1].legend()
    axes[1].set_xlabel("Date", fontsize=14)
    axes[1].set_ylabel("Price", fontsize=14)
    axes[1].set_title(f"{stock_name} Highest vs Lowest Price", fontsize=20)

    plt.tight_layout()
    plt.show()





def plot_stocks_closing_200MA(x, stock_col='stock', date_col='date', closing_col='closing'):
    """
    Plot the closing prices and 200-day moving average for each stock.

    Parameters:
    df (pd.DataFrame): DataFrame containing stock data.
    stock_col (str): Column name for stock identifiers.
    date_col (str): Column name for dates.
    closing_col (str): Column name for closing prices.

    Returns:
    None
    """
    df = x.copy()
    stocks = df.index.unique()
    num_stocks = len(stocks)
    fig, axes = plt.subplots(num_stocks, 1, figsize=(12, 6 * num_stocks))

    if num_stocks == 1:
        axes = [axes]

    for ax, stock in zip(axes, stocks):
        stock_data = df[df.index == stock].sort_values(by=date_col)
        stock_data['200_MA'] = stock_data[closing_col].rolling(window=200).mean()
        stock_data = stock_data.dropna(subset=['200_MA'])

        ax.plot(stock_data[date_col], stock_data[closing_col], label='Closing Price', color='blue')
        ax.plot(stock_data[date_col], stock_data['200_MA'], label='200-Day MA', color='orange')
        ax.set_title(f'{stock} Closing Prices and 200-Day Moving Average')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

    plt.tight_layout()
    plt.show()

def print_strong_corr(df, target_name):
    if isinstance(target_name, str):
        target_name = [target_name]
    cor_lst = df.corr()[target_name[0]]
    for tn in target_name:
        cor_lst = cor_lst.drop(tn)
    strong = [(k, c) for k, c in cor_lst.items() if np.abs(c) > 0.25]
    if len(strong) > 0:
        print(f"Variables with Strong Correlation with variable {target_name[0]} (> 0.25) : ", strong)
    else: print("No Strong Pearson Correlations.")
    return strong, len(strong)


def plot_target(x, classification=True, direction=True, duration=1, corr=True):

    df = x.copy()
    if classification:
        target_name = f'clf_target_{duration}d'
        if direction:
            lbl = "Meaningful up movement: 1"
        else:
            lbl = "Meaningful down movement: 1"
        sns.countplot(data=df, x=target_name, label=lbl)
    else:
        target_name = f'reg_target_{duration}d'
        sns.histplot(data=df, x=target_name, label=target_name)

    plt.title("Target Variable Distribution")
    plt.show()
    if corr:
        print_strong_corr(df, target_name)


def plot_time_since_last(x, corr=True):

    df = x.copy()
    print(f"Maximum idle time between trades is: {int(df['time_since_last'].max())} days.")
    
    sns.histplot(data=df, x='time_since_last')
    plt.title("Time Since Last Trade Count Distribution", fontsize=20)
    plt.show()

    if corr:
        print_strong_corr(df, 'time_since_last')



def plot_stock_ma(x, list=[20, 50, 200], corr=True):
    """
    Plot the closing prices and moving averages created for a single stock.

    Parameters:
    df (pd.DataFrame): DataFrame containing stock data.
    stock_col (str): Column name for stock identifiers.
    date_col (str): Column name for dates.
    closing_col (str): Column name for closing prices.

    Returns:
    None
    """
    for v in list:
        print(f"Moving Average for {v} Days Created.")
    print("MA Ratio Created")
    num_plots = len(list)
    df = x.copy()
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots))

    if num_plots == 1:
        axes = [axes]

    for ax, ma in zip(axes, list):
        
        ax.plot(df.index, df['closing'], label='Closing Price', color='blue')
        ax.plot(df.index, df[f'{ma}_MA'], label=f'{ma}-Days Moving Average', color='orange')

        ax.set_title(f'Closing Price and {ma}-Day Moving Average')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 4))
    sns.histplot(data=df, x='ma_ratio', label= 'Moving Average Ratio', bins=50, kde=True)

    plt.title("Moving Averages Ratio Price vs Long Trend Count Distribution ")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 4))
    sns.histplot(data=df, x='ma_fast_slow', label= 'Moving Average Ratio', bins=50, kde=True)

    plt.title("Moving Averages Ratio Short Trend vs Long Trend Count Distribution ")
    plt.legend()
    plt.show()

    mas = [f'{m}_MA' for m in list]
    print(df[mas].corr())

    if corr:
        print_strong_corr(df, mas)


def plot_range(x, clf_target='clf_target_1d', corr=True):

    df = x.copy()

    if corr:
        lst, num_plots = print_strong_corr(df, 'range')

    if num_plots == 0: return
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots))
    axes = [axes] if num_plots == 1 else axes

    keys = [k for k, _ in lst]
    for ax, k in zip(axes, keys):
        for lbl, color in [(0, 'red'), (1, 'blue')]:
            m = df[clf_target] == lbl
            ax.scatter(df.loc[m, k], df.loc[m, 'range'], color=color, label=str(lbl), alpha=0.35)
        ax.legend(title=clf_target)
        ax.set_title(f'Range vs {k}')
        ax.set_xlabel(k)
        ax.set_ylabel('Range')

    plt.tight_layout()
    plt.show()

def plot_returns(x, list=[1, 3, 5], corr=True):

    df = x.copy()
    df['r1'] = df['closing'].pct_change(1)
    num_plots = len(list)
    for r in list:
        print(f"{r}-Day Return Variable Created.")
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots))

    for ax, r in zip(axes, list):
        ax.plot(df.index, df[f'r{r}'], label=f'{r}-Days Returns', color='orange')
        ax.plot(df.index, df['r1'], label='1-Day Returns', color='blue', alpha=0.7)
        ax.set_title(f'Closing Price and {r}-Day Return')
        ax.set_xlabel('Date')
        ax.set_ylabel('Percentage')
        ax.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    trans = 1
    num_plots += 1
    for r in list:
        plt.hist(df[f'r{r}'], alpha=trans, label=f'r{r}', bins=30)
        trans -= 1/num_plots
    plt.title("Daily Returns Variables Count Distribution")
    plt.show()

    drs = [f'r{day}' for day in list]
    print(df[drs].corr())

    if corr:
        print_strong_corr(df, drs)


def plot_volatility(x, list=[10, 20], threshold=0.025, corr=True):
    for v in list:
        print(f"{v}-Day Volatility Variable Created.")
    df = x.copy()
    for v in list:
        plt.plot(df.index, df[f'volatility_{v}d'], label=f'volatility Over {v}-Days')
    plt.axhline(threshold, linestyle='--', color='red', label="High Volatility Threshold") #For such an illiquidate market
    plt.legend()
    plt.show() 


    vols = [f'volatility_{day}d' for day in list]
    print(df[vols].corr())

    if corr:
        print_strong_corr(df, vols)

def plot_rsi(x, list=[7, 14], corr=True):
    df = x.copy()

    for r in list:
        print(f"RSI for {r}- days created.")
    num_plots = len(list)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots))

    if num_plots == 1:
        axes = [axes]
    
    RSIs = [f'rsi_{v}' for v in list]
    for ax, r, d in zip(axes, RSIs, list):
        ax.plot(df.index, df[r], label='RSI for {d} Days')
        ax.axhline(75, label='Overbought', color='Green')
        ax.axhline(25, label='Oversold', color='Red')

    plt.tight_layout()
    plt.show()

    print(df[RSIs].corr())
    if corr:
        print_strong_corr(df, RSIs)


def plot_liquidity(x):
    df = x.copy()

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['n_deals_change'], label='Change in # Deals', color='orange')
    plt.axhline(0, linestyle='--', alpha=0.6)
    plt.title("Change in Number of Deals Over Time")
    plt.xlabel("Date")
    plt.ylabel("Percentage Change")
    plt.legend()
    plt.show()


    plt.figure(figsize=(8, 4))
    plt.hist(df['n_deals_change'].dropna(), bins=40)
    plt.title("Distribution of Change in Number of Deals")
    plt.xlabel("Percentage Change")
    plt.ylabel("Count")
    plt.show()

def plot_macd(x):
    df = x.copy()

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['macd'], label='MACD', color='blue')
    plt.plot(df.index, df['macd_signal'], label='Signal', color='orange')
    plt.axhline(0, linestyle='--', alpha=0.6)

    plt.title("MACD and Signal Line")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


    plt.figure(figsize=(12, 4))
    plt.bar(df.index, df['macd_hist'], width=1.0)
    plt.axhline(0, linestyle='--', alpha=0.6)

    plt.title("MACD Histogram")
    plt.xlabel("Date")
    plt.ylabel("Histogram Value")
    plt.show()

def plot_stochastic(x):
    df = x.copy()

    plt.figure(figsize=(12, 4))


    plt.plot(df.index, df['stoch_d'], label='%D (Slow)', color='orange')


    plt.plot(df.index, df['stoch_diff'], label='%K - %D (Diff)', color='blue', alpha=0.7)


    plt.axhline(80, linestyle='--', color='red', alpha=0.6, label='Overbought (80)')
    plt.axhline(20, linestyle='--', color='green', alpha=0.6, label='Oversold (20)')

    plt.axhline(0, linestyle='--', color='black', alpha=0.5, label='Diff = 0 (Crossover)')

    plt.title("Stochastic Oscillator: %D and Diff (%K-%D)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
