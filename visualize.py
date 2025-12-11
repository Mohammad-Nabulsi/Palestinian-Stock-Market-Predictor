import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distributions(df, feature_cols, target_col=None):
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

def plot_stocks_closing_200MA(df, stock_col='stock', date_col='date', closing_col='closing'):
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
    stocks = df[stock_col].unique()
    num_stocks = len(stocks)
    fig, axes = plt.subplots(num_stocks, 1, figsize=(12, 6 * num_stocks))

    if num_stocks == 1:
        axes = [axes]

    for ax, stock in zip(axes, stocks):
        stock_data = df[df[stock_col] == stock].sort_values(by=date_col)
        stock_data['200_MA'] = stock_data[closing_col].rolling(window=200).mean()

        ax.plot(stock_data[date_col], stock_data[closing_col], label='Closing Price', color='blue')
        ax.plot(stock_data[date_col], stock_data['200_MA'], label='200-Day MA', color='orange')
        ax.set_title(f'{stock} Closing Prices and 200-Day Moving Average')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

    plt.tight_layout()
    plt.show()