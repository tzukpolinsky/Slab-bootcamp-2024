import pandas as pd
import statsmodels as sm
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


def anova_fdr(data_set: pd.DataFrame, cat_var: str, cont_var: str):
    """
    Do some stats on two different types of data and fix the results.

    Parameters:
    data (pd.DataFrame): The input data frame.
    cat_var (str): The name of the categorical variable.
    cont_var (str): The name of the continuous variable.

    Returns:
    tuple: ANOVA test results, pairwise comparison results, FDR adjusted p-values.
    """
    # Define the mdl formula
    formula = f'{cont_var} ~ C({cat_var})'

    # Fit the mdl
    mdl = ols(formula, data=data_set).fit()

    # Perform ANOVA
    anova_table = sm.stats.anova_lm(mdl, typ=2)

    # Perform pairwise comparisons using Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(endog=data_set[cont_var], groups=data_set[cat_var], alpha=0.05)

    # Extract p-values from the pairwise comparison results
    p_ress = tukey_result.pvalues

    # Perform FDR adjustment on the p-values
    _, pvals_corrected, _, _ = multipletests(p_ress, alpha=0.05, method='fdr_bh')
    tukey_result_data = np.array(tukey_result._results_table.data)
    # Create a DataFrame for the pairwise comparison results with FDR adjusted p-values
    pairwise_results = pd.DataFrame(data={
        'group1': tukey_result_data[1:, 0],
        'group2': tukey_result_data[1:, 1],
        'meandiff': tukey_result_data[1:, 2],
        'p-adj': tukey_result_data[1:, 3],
        'lower': tukey_result_data[1:, 4],
        'upper': tukey_result_data[1:, 5],
        'reject': tukey_result_data[1:, 6],
        'pvals_corrected': pvals_corrected
    })

    return anova_table, pairwise_results


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


def norm_test_res(is_normal: bool, JB: float, p_res: float, skewness: float, kurtosis: float,
                  alpha: float, skewness_threshold: float, kurtosis_limits: Tuple[float, float]):
    """
    Print the conclusion from the normality test.
    """
    if is_normal:
        print("Conclusion: The residuals appear to be normally distributed.")
    else:
        print("Conclusion: The residuals do not appear to be normally distributed.")
        if p_res <= alpha:
            print(f"  - The Jarque-Bera test indicates non-normality (p-value <= {alpha}).")
        if abs(skewness) >= skewness_threshold:
            print(f"  - The distribution is skewed (|skewness| >= {skewness_threshold}).")
        if kurtosis <= kurtosis_limits[0] or kurtosis >= kurtosis_limits[1]:
            print(f"  - The distribution has abnormal kurtosis (outside range {kurtosis_limits}).")


def multi_coll_res(no_multicollinearity: bool, threshold: float):
    """
    Print the conclusion from the multicollinearity test.
    """
    if no_multicollinearity:
        print("Conclusion: No multicollinearity detected.")
    else:
        print("Conclusion: Multicollinearity detected.")
        print(f"  Features with VIF > {threshold:.1f} may be problematic.")


def test_feat_combos(data_set, feature_columns, target_column, num_features=2, with_conclusion_print=False):
    for feature_combo in combinations(feature_columns, num_features):
        inp = data_set[list(feature_combo)]
        y = data_set[target_column]
        mdl = sm.OLS(y, sm.add_constant(inp.to_numpy())).fit()
        residuals = np.array(mdl.resid)
        y_pred = mdl.predict(sm.add_constant(inp.to_numpy()))
        is_normal, JB, p_res, skewness, kurtosis = check_normality(residuals,
                                                                   with_conclusion_print=with_conclusion_print)
        plot_normality_test(residuals, feature_combo, target_column, is_normal, JB, p_res, skewness, kurtosis)
        is_homoscedastic, lm, lm_pvalue, fvalue, f_pvalue = check_homoscedasticity(y, y_pred,
                                                                                   with_conclusion_print=with_conclusion_print)
        homo_test_plot(y, y_pred, feature_combo, target_column, is_homoscedastic, lm, lm_pvalue, fvalue, f_pvalue)

        is_linear, lin_p_res, fstat = check_linearity(inp, y, with_conclusion_print=with_conclusion_print)
        plot_linearity_test(inp, y.values, feature_combo, target_column, is_linear, lin_p_res, fstat)

        no_multicollinearity, vif_values = multi_coll_check(inp, with_conclusion_print=with_conclusion_print)
        plot_multicollinearity_test(vif_values, threshold=5.0)

        no_autocorrelation, lb_p_res, dw_statistic = check_autocorrelation(residuals,
                                                                           with_conclusion_print=with_conclusion_print)
        auto_corr_plot(residuals, feature_combo, target_column, no_autocorrelation, lb_p_res, dw_statistic)


def multi_coll_check(inp: pd.DataFrame, threshold: float = 5.0, with_conclusion_print=False) -> Tuple[
    bool, List[float]]:
    """
    Check for multicollinearity using Variance Inflation Factor (VIF).

    Args:
        inp (pd.DataFrame): Feature matrix.
        threshold (float): VIF threshold for multicollinearity. Default is 5.0.
        with_conclusion_print (bool): print the conclusion of the test.

    Returns:
        Tuple[bool, List[float]]: A tuple containing:
            - bool: True if no multicollinearity detected (all VIF values < threshold), False otherwise.
            - List[float]: List of VIF values for each feature.

    Reference:
    Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2005). Applied Linear
    Statistical Models (5th ed.). McGraw-Hill/Irwin.
    """
    inp_with_const = sm.add_constant(inp)
    vif_values = [variance_inflation_factor(inp_with_const.values, i) for i in range(1, inp_with_const.shape[1])]
    is_multicollinearity = all(vif < threshold for vif in vif_values)
    if with_conclusion_print:
        multi_coll_res(is_multicollinearity, threshold)
    return is_multicollinearity, vif_values
