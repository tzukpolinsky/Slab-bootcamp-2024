import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
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