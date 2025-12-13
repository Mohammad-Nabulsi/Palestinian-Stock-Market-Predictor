import numpy as np
import pandas as pd


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