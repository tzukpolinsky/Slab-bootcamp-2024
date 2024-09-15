
def multi_var_regr(data_set: pd.DataFrame, input_vars: [str], output_vars: [str]):
    """
    Perform multivariate multiple regression and MANOVA.

    Args:
    data_set (pd.DataFrame): Holds some information, kind of important.
    input_vars (list): List of column names for independent variables
    output_vars (list): List of column names for dependent variables

    Returns:
    tuple: (mdl, manova_results)
        mdl: The fitted OLS mdl
        manova_results: Dictionary containing MANOVA results,
        mean_rsquared: the mean rsquared for each inp predict 1 y column
    """
    inp = data_set[input_vars]
    if len(inp) == 0:
        print("inp is empty")
        return None, 0.5, 0
    inp = sm.add_constant(inp)
    out = data_set[output_vars]
    if len(out) == 0:
        print("out is empty")
        return None, 0.5, 0
    rsquared_values = []
    for y_col in output_vars:
        mdl = sm.OLS(out[y_col], inp).fit()
        rsquared_values.append(mdl.rsquared)
    mdl = sm.OLS(out, inp).fit()
    formula = ' + '.join(output_vars) + ' ~ ' + ' + '.join(input_vars)
    manova = MANOVA.from_formula(formula, data=data_set)
    manova_results = manova.mv_test()
    manova_p_res = manova_results.results['Intercept']['stat']['Pr > F']['Pillai\'s trace']

    return mdl, manova_p_res, sum(rsquared_values) / len(rsquared_values) if len(rsquared_values) > 0 else 0
