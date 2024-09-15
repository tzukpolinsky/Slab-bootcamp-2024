def manova_eta_fdr(data_tbl: pd.DataFrame, dep_vars: [str], cat_vars: [str]):
    """
    Do a multi-test, adjust stuff, and get some squared number at the end.

    Parameters:
    data (pd.DataFrame): The input data frame.
    dep_vars (list of str): The names of the dependent variables.
    cat_vars (list of str): The names of the categorical variables.

    Returns:
    tuple: MANOVA results, ANOVA test results with eta squared, pairwise comparison results with FDR adjusted p-values.
    """
    categorical_frm = ' + '.join(cat_vars)
    frm = f'{"+".join(dep_vars)} ~ {categorical_frm}'
    manova = MANOVA.from_frm(frm, data=data_tbl)

    stat_res = {}
    pairwise_results = {}
    eta_num_results = {}

    for dependent_var in dep_vars:
        # Define the ANOVA frm for each dependent variable
        anova_frm = f'{dependent_var} ~ {categorical_frm}'

        # Fit the ANOVA model
        model = ols(anova_frm, data=data_tbl).fit()

        # Perform ANOVA
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Calculate eta squared
        ss_effect = anova_table['sum_sq'].iloc[0]
        ss_total = anova_table['sum_sq'].sum()
        eta_num = ss_effect / ss_total

        # Add eta squared to the ANOVA table
        anova_table['eta_sq'] = [eta_num] + [None] * (len(anova_table) - 1)

        # Store ANOVA table
        stat_res[dependent_var] = anova_table

        # Perform pairwise comparisons using Tukey's HSD test
        tukey_result = pairwise_tukeyhsd(endog=data_tbl[dependent_var], groups=data_tbl[cat_vars[0]], alpha=0.05)

        # Extract p-values from the pairwise comparison results
        p_values = tukey_result.pvalues

        # Perform FDR adjustment on the p-values
        _, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        tukey_result_data = np.array(tukey_result._results_table.data)
        # Create a DataFrame for the pairwise comparison results with FDR adjusted p-values
        pairwise_result = pd.DataFrame(data={
            'group1': tukey_result_data[1:, 0],
            'group2': tukey_result_data[1:, 1],
            'meandiff': tukey_result_data[1:, 2],
            'p-adj': tukey_result_data[1:, 3],
            'lower': tukey_result_data[1:, 4],
            'upper': tukey_result_data[1:, 5],
            'reject': tukey_result_data[1:, 6],
            'pvals_corrected': pvals_corrected
        })

        # Store pairwise results
        pairwise_results[dependent_var] = pairwise_result

    return manova, stat_res, pairwise_results
