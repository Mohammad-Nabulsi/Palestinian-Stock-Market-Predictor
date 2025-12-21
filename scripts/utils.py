import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

from scripts import create_features as cf
from scripts import visualize as vz

importlib.reload(cf)
importlib.reload(vz)



def take_stock(x, stock_name, plot=True):
    """
        it slices a stock from all stocks data frame and sets it's date as the index
    """
    df = x.loc[stock_name, :].copy()
    df.set_index('date', inplace=True)
    df.sort_index(ascending=True, inplace=True)

    if plot:
        print(df.head())
        vz.plot_initial(df, stock_name)
    return df

def nyears(data, n=4):
    df = data.copy()
    df_nyears = df[df.index >= (df.index.max() - pd.DateOffset(years=n))]

    #vz.plot_initial(df_nyears)
    return df_nyears

FEATURE_REGISTRY = {

    "time_since_last": {
        "create": cf.create_time_since_last,
        "plot": vz.plot_time_since_last,
        "create_params": {},
        "plot_params": {"corr": True}
    },

    "target": {
        "create": cf.create_target_variable,
        "plot": vz.plot_target,
        "create_params": {
            "classification": True,
            "duration": 1,
            "direction": True,
            "epsilon": 0.001
        },
        "plot_params": {
            "classification": True,
            "duration": 1,
            "direction": True,
            "corr": True
        }
    },

    "moving_averages": {
        "create": cf.create_moving_averages,
        "plot": vz.plot_stock_ma,
        "create_params": {"list": [20, 50, 200]},
        "plot_params": {"list": [20, 50, 200], "corr": True}
    },

    "range": {
        "create": cf.create_range,
        "plot": vz.plot_range,
        "create_params": {},
        "plot_params": {
            "clf_target": "clf_target_1d",
            "corr": True
        }
    },

    "returns": {
        "create": cf.create_daily_returns,
        "plot": vz.plot_returns,
        "create_params": {"list": [1, 3, 5]},
        "plot_params": {
            "list": [1, 3, 5],
            "corr": True
            }
    },

    "volatility": {
        "create": cf.create_volatility,
        "plot": vz.plot_volatility,
        "create_params": {"list": [10, 20]},
        "plot_params": {
            "list": [10, 20],
            "corr": True
            }
    },

    "rsi": {
        "create": cf.create_rsi,
        "plot": vz.plot_rsi,
        "create_params": {"list": [7, 14]},
        "plot_params": {
            "list": [7, 14],
            "corr": True
            }
    },
    "liquidity": {
        "create": cf.create_liquidity,
        "plot": vz.plot_liquidity,
        "create_params": {},
        "plot_params": {}
    },
    "macd": {
        "create": cf.create_macd,
        "plot": vz.plot_macd,
        "create_params": {},
        "plot_params": {}        
    },
    "stochastic": {
        "create": cf.create_stochastic,
        "plot": vz.plot_stochastic,
        "create_params": {"window": 14},
        "plot_params": {}          
    }
}


def create_and_plot(x, name, plot=True, **override_params):
    if name not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature: {name}")

    cfg = FEATURE_REGISTRY[name]
    df = x.copy()

    # -------- CREATE --------
    create_params = cfg.get("create_params", {}).copy()
    for k, v in override_params.items():
        if k in create_params:
            create_params[k] = v

    df = cfg["create"](df, **create_params)

    # -------- PLOT --------
    if plot and "plot" in cfg:
        plot_params = cfg.get("plot_params", {}).copy()
        for k, v in override_params.items():
            if k in plot_params:
                plot_params[k] = v

        cfg["plot"](df, **plot_params)
    #Special case cause we need the dropped column in plotting
    if name == 'macd':
        df.drop(columns="macd_signal", inplace=True)

    return df

def normalize_range(x):
    df = x.copy()
    df['range_norm'] = df['range'] / df['closing']
    print("Normalized Range Created")
    print(df[['range', 'range_norm']].corr())
    print("Due to almost perfect correlation range feature dropped.")
    df.drop(columns='range', inplace=True)
    return df

def add_temporal_features(x):
    df = x.copy()
    df['day_of_week'] = df.index.dayofweek
    df['first_week_of_month'] = (df.index.day <= 7).astype(int)

    print("Temporal Features Created.")

    return df

