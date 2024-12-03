def homo_check(data_tbl: pd.DataFrame, y: pd.Series, alpha=0.05, with_conclusion_print=False) -> Tuple[
    bool, float, float, float, float]:
    """
    See if the spread of data is even by running a specific test on it.

    This function performs the Breusch-Pagan test for heteroscedasticity. It uses different
    criteria based on the sample size to determine homoscedasticity.

    Args:
        data_tbl (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        alpha (float, optional): The significance level for the test. Defaults to 0.05.
        with_conclusion_print (bool): print the conclusion of the test.
    Returns:
        Tuple[bool, float, float, float, float]: A tuple containing:
            - bool: True if variance is likely homoscedastic, False otherwise.
              For sample sizes <= 30, only the F-test p-value is considered.
              For larger samples, both LM and F-test p-values must exceed alpha.
            - float: Lagrange Multiplier (LM) statistic
            - float: p-value for the LM statistic
            - float: F-value
            - float: p-value for the F-statistic

    Notes:
        - For small samples (n <= 30), only the F-test is used due to its better small-sample properties.
        - For larger samples, both tests must indicate homoscedasticity for the function to return True.

    Reference:
    Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and
    random coefficient variation. Econometrica, 47(5), 1287-1294.
    https://www.jstor.org/stable/1911963
    """
    input_with_const = sm.add_constant(data_tbl)

    # Fit the model
    model = sm.OLS(y, input_with_const).fit()
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
    if len(data_tbl) <= 30:
        is_homoscedasticity = lm_pvalue > alpha and f_pvalue > alpha

    else:
        is_homoscedasticity = lm_pvalue > alpha and f_pvalue > alpha
    if with_conclusion_print:
        print_homoscedasticity_conclusion(is_homoscedasticity, lm_pvalue, f_pvalue, alpha, len(data_tbl))
    return is_homoscedasticity, lm, lm_pvalue, fvalue, f_pvalue