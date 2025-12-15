import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

from scripts import create_features as cf
from scripts import visualize as vz

importlib.reload(cf)
importlib.reload(vz)


def moving_averages(x, list=[20, 50, 200]):
    """
    x: must be an individual stock dataframe
    will create the passed or default values and visualzie it 
    Docstring for create_moving_averages
    
    :param df: Description
    :param list: Description
    """

    df = x.copy()
    df = cf.create_moving_averages(df, list)

    vz.plot_stock_ma(df, list)



    return df

def range(x):
    df = x.copy()
    df = cf.create_range(df)

    vz.plot_range(df)

    return df

def returns(x, list=[1, 3, 5]):

    df = x.copy()
    df = cf.create_daily_returns(df)
    vz.plot_returns(df)
    return df

def volatility(x, list=[10, 20, 30]):
    df = x.copy()
    df = cf.create_volatitlity(df, list)
    vz.plot_volatility(df)
    return df