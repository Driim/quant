import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import numpy as np
import pandas as pd

# Augmented Dickey-Fuller
def do_adf_regression(df, ticker1, ticker2) -> int:
    x = df[ticker1].values
    y = df[ticker2].values

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()

    alpha, beta = model.params
    residuals = y - (alpha + beta * x[:, 1])

    if np.isnan(residuals).any():
        return np.nan

    t_stat, p_value, *rest = adfuller(model.resid)
    critical_values = rest[2]
    probability = np.nan
    if p_value < 0.01 and t_stat < critical_values['1%']:
        probability = 99
    elif p_value < 0.05 and t_stat < critical_values['5%']:
        probability = 95
    elif p_value < 0.1 and t_stat < critical_values['10%']:
        probability = 90

    return probability


# Augmented Engle-Granger
def do_aeg_regression(df, ticker1, ticker2) -> int:
    x = df[ticker1].values
    y = df[ticker2].values

    t_stat, p_value, critical_values = coint(x, y)

    probability = np.nan
    if p_value < 0.01 and t_stat < critical_values[0]:
        probability = 99
    elif p_value < 0.05 and t_stat < critical_values[1]:
        probability = 95
    elif p_value < 0.1 and t_stat < critical_values[2]:
        probability = 90

    return probability