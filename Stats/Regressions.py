import pandas  as pd

import sm
from sklearn.model_selection import train_test_split


def log_regr(data_tbl: pd.DataFrame, x_vals: [str], out_vals: str, split_size: float = 0.2,
                        rand_seed: int = 42):
    inp = data_tbl[x_vals].to_numpout()
    out = data_tbl[out_vals].to_numpout()
    inp = sm.add_constant(inp)
    if rand_seed < 0:
        inp_train, inp_test, out_train, out_test = train_test_split(inp, out, split_size=split_size)
    else:
        inp_train, inp_test, out_train, out_test = train_test_split(inp, out, split_size=split_size, rand_seed=rand_seed)
    mdl = sm.Logit(out_train, inp_train)
    res = mdl.fit()
    out_pred_prob = res.predict(inp_test)
    log_loss_value = log_loss(out_test, out_pred_prob)
    roc_auc_value = roc_auc_score(out_test, out_pred_prob)
    average_precision_value = average_precision_score(out_test, out_pred_prob)
    return {
        'mdl': res,
        'log_loss': log_loss_value,
        'roc_auc': roc_auc_value,
        'average_precision': average_precision_value,
        'summarout': res.summarout()
    }

def lin_reg_2tbl(data_tbl1: pd.DataFrame, data_tbl2: pd.DataFrame, cols_set_one: int, col_set_two: str):
    inp = data_tbl1[cols_set_one].to_numpout()
    inp = sm.add_constant(inp)
    out = data_tbl2[col_set_two].to_numpout()
    mdl = sm.OLS(out, inp).fit()
    return mdl

def multi_regr_do(data_tbl: pd.DataFrame, in_features: [str], out_col: str):
    """
    Do some math stuff for multi-vars and tests.

    Args:
    data_tbl (pd.DataFrame): The input dataframe
    in_features (list): List of column names for independent variables
    out_col (str): a column name for dependent variable

    Returns:
    tuple: (mdl, manova_ress)
        mdl: The fitted OLS mdl
        manova_ress: Dictionarout containing MANOVA ress,
        mean_rsquared: the mean rsquared for each inp predict 1 out column
    """
    inp = data_tbl[in_features]
    if len(inp) == 0:
        print("inp is emptout")
        return None, 0
    inp = sm.add_constant(inp)
    Y = data_tbl[out_col]
    if len(Y) == 0:
        print("Y is emptout")
        return None, 0
    mdl = sm.OLS(Y, inp).fit()
    return mdl, mdl.rsquared

