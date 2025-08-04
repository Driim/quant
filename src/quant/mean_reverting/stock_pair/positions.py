import pandas as pd
import numpy as np

def calculate_positions(data: pd.DataFrame, ticker1: str, ticker2: str, open_threshold: float, close_threshold: float) -> list[pd.Series]:
    data['pos_1_long'] = 0
    data['pos_2_long'] = 0
    data['pos_1_short'] = 0
    data['pos_2_short'] = 0
    data.loc[data['zscore'] >= open_threshold, ('pos_1_short', 'pos_2_short')] = [ -1, 1 ] # Short spread
    data.loc[data['zscore'] <= -open_threshold, ('pos_1_long', 'pos_2_long')] = [ 1, -1 ] # Long spread
    data.loc[data['zscore'] <= close_threshold, ('pos_1_short', 'pos_2_short')] = 0 # Close position short
    data.loc[data['zscore'] >= -close_threshold, ('pos_1_long', 'pos_2_long')] = 0 # Close position long

    data.ffill(inplace=True)

    longs = data.loc[:, ('pos_1_long', 'pos_2_long')]
    shorts = data.loc[:, ('pos_1_short', 'pos_2_short')]

    positions = np.array(longs) + np.array(shorts)
    positions = pd.DataFrame(positions)

    longs_with_dates = data.loc[:, ('pos_1_long', 'pos_2_long')].copy()
    longs_with_dates.columns = [ticker1, ticker2]

    shorts_with_dates = data.loc[:, ('pos_1_short', 'pos_2_short')].copy()
    shorts_with_dates.columns = [ticker1, ticker2]

    positions_with_dates = longs_with_dates + shorts_with_dates

    data.drop(columns=['pos_1_long', 'pos_2_long', 'pos_1_short', 'pos_2_short'], inplace=True)

    return (positions, positions_with_dates)