from typing import Callable
import pandas as pd
import numpy as np

def calculate_positions_from_lambda(data: pd.DataFrame, long: Callable[[any], pd.Series], short: Callable[[any], pd.Series]) -> pd.DataFrame:
    df = data.copy()
    
    df[['pos_1_short', 'pos_2_short']] = df.apply(lambda row: short(row), axis=1) # Short spread
    df[['pos_1_long', 'pos_2_long']] = df.apply(lambda row: long(row), axis=1)    # Long spread

    # Everything that isn't a signal will be NAN, so we fill data in right way
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    longs = df.loc[:, ('pos_1_long', 'pos_2_long')].copy()
    longs.columns = ['ticker1', 'ticker2']

    shorts = df.loc[:, ('pos_1_short', 'pos_2_short')].copy()
    shorts.columns = ['ticker1', 'ticker2']

    positions = longs + shorts
    return positions

def calculate_positions_from_zscore(data: pd.DataFrame, open_threshold: float, close_threshold: float) -> pd.DataFrame:
    def short(row) -> pd.Series:
        if (row['zscore'] >= open_threshold):
            return pd.Series([-1, 1])
        elif (row['zscore'] <= close_threshold):
            return pd.Series([0, 0])
        
        return pd.Series([np.nan, np.nan])
    
    def long(row) -> pd.Series:
        if (row['zscore'] <= -open_threshold):
            return pd.Series([1, -1])
        elif (row['zscore'] >= - close_threshold):
            return pd.Series([0,0])
        
        return pd.Series([np.nan, np.nan])
    
    return calculate_positions_from_lambda(data, long, short)