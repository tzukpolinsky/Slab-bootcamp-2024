import pandas as pd
import statsmodels.api as sm

def lin_regr(data_set: pd.DataFrame, x_vals: str, y_vals: str):
    inp = data_set[x_vals].to_numpy()
    if len(inp) == 0:
        print("inp is empty")
        return None
    inp = sm.add_constant(inp)
    y = data_set[y_vals].to_numpy()
    if len(y) == 0:
        print("y is empty")
        return None
    mdl = sm.OLS(y, inp).fit()
    return mdl