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


def prep_data(data_set, bool_cat_feats, cont_vars):
    """
    Fix the data so it can be used by messing with the columns and scaling some numbers.
    Replace original columns with transformed data in a copy of the original DataFrame.

    Args:
    data_set (pd.DataFrame): Holds some information, kind of important.
    bool_cat_feats (list): These are some variables, not sure what kind.
    cont_vars (list): More variables, probably numbers.

    Returns:
    pd.DataFrame: Copy of original dataframe with preprocessed data
    """
    # Create a copy of the original DataFrame
    data_set_copy = data_set.copy()

    # Create preprocessing steps
    proc = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), cont_vars),
            ('cat', OneHotEncoder(drop='first'), bool_cat_feats)
        ])

    # Fit and transform the data
    preprocessed_data = proc.fit_transform(data_set_copy)

    # Get feature names after preprocessing
    onehot_cols = proc.named_transformers_['cat'].get_feature_names_out(bool_cat_feats)
    feature_names = list(cont_vars) + list(onehot_cols)

    # Create a new dataframe with processed data
    data_set_processed = pd.DataFrame(preprocessed_data, columns=feature_names, index=data_set_copy.index)

    # Replace original columns with preprocessed data
    data_set_copy.drop(columns=bool_cat_feats + cont_vars, inplace=True)
    data_set_copy = pd.concat([data_set_copy, data_set_processed], axis=1)

    return data_set_copy


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
        plot_normality_test(residuals,feature_combo,target_column ,is_normal, JB, p_res, skewness, kurtosis)
        is_homoscedastic, lm, lm_pvalue, fvalue, f_pvalue = check_homoscedasticity(y, y_pred,
                                                                                   with_conclusion_print=with_conclusion_print)
        homo_test_plot(y, y_pred,feature_combo,target_column, is_homoscedastic, lm, lm_pvalue, fvalue, f_pvalue)

        is_linear, lin_p_res, fstat = check_linearity(inp, y, with_conclusion_print=with_conclusion_print)
        plot_linearity_test(inp, y.values,feature_combo, target_column, is_linear, lin_p_res, fstat)

        no_multicollinearity, vif_values = multi_coll_check(inp, with_conclusion_print=with_conclusion_print)
        plot_multicollinearity_test(vif_values, threshold=5.0)

        no_autocorrelation, lb_p_res, dw_statistic = check_autocorrelation(residuals,
                                                                             with_conclusion_print=with_conclusion_print)
        auto_corr_plot(residuals,feature_combo,target_column, no_autocorrelation, lb_p_res, dw_statistic)

def time_comp_plot(data_set: pd.DataFrame, category_col: str, value_col: str, time_col: str, title: str):
    """
    Create a line plot showing the change in values for each category between two time points.

    Parameters:
    data_set (pandas.DataFrame): The input DataFrame containing the data.
    category_col (str): The name of the column containing category labels (e.g., subject names).
    value_col (str): The name of the column containing the values to be plotted.
    title (str): The title for the plot.

    Returns:
    matplotlib.figure.Figure: The created figure object.

    The function expects the DataFrame to have the following structure:
    - A column for categories (e.g., subject names)
    - A column for time points (assumed to have two unique values)
    - A column for values

    Example usage:
    >>> import pandas as pd
    >>> data = {
    ...     'Subject': ['A', 'B', 'C', 'A', 'B', 'C'],
    ...     'Time': [1, 1, 1, 2, 2, 2],
    ...     'Value': [10, 15, 8, 12, 14, 10]
    ... }
    >>> data_set = pd.DataFrame(data)
    >>> time_comp_plot(data_set, 'Subject', 'Value', 'Change in Values')
    """
    plt.figure(figsize=(10, 6))
    plot_data_set = data_set.copy()
    plot_data_set = plot_data_set.sort_values(by=value_col)
    # Create the lineplot
    sns.lineplot(data=plot_data_set, x=time_col, y=value_col, hue=category_col, marker='o')

    # Customize the plot
    plt.title(title)
    plt.xlabel('Time Point')
    plt.ylabel(value_col)

    # Add value labels
    for line in plt.gca().lines:
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            plt.text(x, y, f' {y:.1f}', va='center', ha='left')
    plt.legend().remove()
    plt.tight_layout()
    plt.show()

def hist_plot(data_set: pd.DataFrame, column: str, bins: int = 10, title: str = None, xlabel: str = None,
                   ylabel: str = 'Frequency') -> np.array:
    """
    Make a bar graph of one column's values and show the frequency.

    This function takes a pandas DataFrame and a column name, and plots a histogram of the data in that column.
    Additional optional parameters allow customization of the plot, including the number of bins, title, and axis labels.

    Parameters:
    data_set (pd.DataFrame): The input DataFrame containing the data.
    column (str): The name of the column to plot the histogram for.
    bins (int, optional): Number of bins for the histogram. Default is 10.
    title (str, optional): The title of the histogram. Default is None.
    xlabel (str, optional): The label for the x-axis. Default is None.
    ylabel (str, optional): The label for the y-axis. Default is 'Frequency'.

    Returns:
    the historgam bins in a np.array

    Raises:
    KeyError: If the specified column does not exist in the DataFrame.
    TypeError: If the input DataFrame is not a pandas DataFrame.

    Example:
    >>> import pandas as pd
    >>> data_set = pd.DataFrame({
    >>>     'age': [23, 45, 56, 67, 34, 45, 56, 78, 89, 34, 23, 45, 56, 67, 78]
    >>> })
    >>> hist_plot(data_set, 'age', bins=5, title='Age Distribution', xlabel='Age', ylabel='Count')
    """
    if not isinstance(data_set, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")

    if column not in data_set.columns:
        raise KeyError(f"The column '{column}' does not exist in the DataFrame.")

    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(data_set[column].dropna(), bins=bins, edgecolor='black')
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    return bins
def cat_hist_plot(data_set: pd.DataFrame, category_column: str, column: str, bins: int = 10, title: str = None,
                            xlabel: str = None,
                            ylabel: str = 'Frequency', amount_of_columns_per_row=4) -> np.array:
    """
    Plot grouped histograms for specified columns in a pandas DataFrame.

    Parameters:
    --------
    - data_set (pd.DataFrame): The input DataFrame containing the data.
    - columns (list): The list of columns to plot grouped histograms for.
    - bins (int, optional): Number of bins for the histogram. Default is 10.
    - title (str, optional): The title of the histogram. Default is None.
    - xlabel (str, optional): The label for the x-axis. Default is None.
    - ylabel (str, optional): The label for the y-axis. Default is 'Frequency'.

    Returns:
    --------
    the bins of the histogram
    """
    if not isinstance(data_set, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")
    if column not in data_set.columns:
        raise KeyError(f"The column '{column}' does not exist in the DataFrame.")
    amount_of_categories = len(data_set[category_column].unique())

    fig, axes = plt.subplots(amount_of_categories // amount_of_columns_per_row + 1,
                             amount_of_categories % amount_of_columns_per_row, figsize=(10, 10))
    all_bins = []
    pal = None
    if category_column in palettes:
        pal = palettes[category_column]
    for i, (category, group_data) in enumerate(data_set.groupby(category_column)):
        color = 'k' if pal is None else pal[category]
        data = group_data[column].dropna()
        n, bins, patches = axes[i].hist(data, bins=bins, alpha=0.2, edgecolor='black',
                                        label=category, color=color, density=True)
        xmin, xmax = axes[i].get_xlim()
        mu, std = norm.fit(data)
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdata_set(x, mu, std)
        axes[i].plot(x, p, color=color, linewidth=2)
        all_bins.append(bins)
        if xlabel:
            axes[i].set_xlabel(xlabel)
        if title:
            axes[i].set_title(title)
        else:
            axes[i].set_title(f'{category}, N:{len(group_data)}')
    fig.tight_layout()
    fig.suptitle(f'{column} Histogram')
    plt.show()
    return all_bins
def regr_res_plot(mdl, input_vars, output_vars, manova_p_res, mean_rsquared):
    """
    Plot actual vs predicted values and residuals for each dependent variable.

    Args:
    mdl: The fitted OLS mdl
    input_vars (list): List of column names for independent variables
    output_vars (list): List of column names for dependent variables
    manova_p_res (float): Overall p-value from MANOVA
    manova_p_res (float): mean of the resquared of each y col with it's inp cols
    """
    n_cols = len(output_vars)
    n_rows = len(input_vars)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 12))

    out = mdl.mdl.endog
    inp = mdl.mdl.exog
    out_pred = mdl.predict()

    for i, col in enumerate(output_vars):
        for j, col2 in enumerate(input_vars):
            ax = axes[j, i]
            ax.scatter(out[:, i], inp[:, j], alpha=0.5)
            ax.plot([out_pred[:, i].min(), out_pred[:, i].max()], [out_pred[:, i].min(), out_pred[:, i].max()], 'r--', lw=2)
            ax.set_xlabel(f'{col} Values')
            ax.set_ylabel(f'{col2} Values')

    manova_sig = "Significant" if manova_p_res < 0.05 else "Not Significant"
    plt.suptitle(
        f'{" ".join(input_vars)} vs {" ".join(output_vars)}\nMANOVA: {manova_sig} (p = {manova_p_res:.3f}),mean R² = {mean_rsquared:.3f}',
        fontsize=16)
    plt.tight_layout()
    plt.show()

def homo_test_plot(y_true: np.ndarray, y_pred: np.ndarray, feature_combo: [str], target_column: str,
                               is_homoscedastic: bool, lm: float,
                               lm_pvalue: float, fvalue: float, f_pvalue: float):
    """
    Plot the results of the homoscedasticity test.
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.suptitle(
        f'Residuals vs Fitted Values (LM={lm:.2f}, p={lm_pvalue:.4f}, F={fvalue:.2f}, p={f_pvalue:.4f}) results are: {"homoscedastic" if is_homoscedastic else "not homoscedastic"}')
    plt.title(f'[{",".join(feature_combo)}] vs {target_column}')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()
def auto_corr_plot(residuals: np.ndarray,feature_combo:[str],target_column:str, no_autocorrelation: bool, lb_p_res: float, dw_statistic: float):
    """
    Plot the results of the autocorrelation test.
    """
    from statsmdls.graphics.tsaplots import plot_acf

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Residuals plot
    ax1.plot(residuals)
    ax1.set_title('Residuals Over Time')
    ax1.set_xlabel('Observation')
    ax1.set_ylabel('Residual')

    # Autocorrelation plot
    plot_acf(residuals, ax=ax2, lags=40)
    ax2.set_title(
        f'Autocorrelation (LB p={lb_p_res:.4f}, DW={dw_statistic:.2f}) results are: {"no autocorrelation" if no_autocorrelation else "autocorrelation"}')
    plt.title(f'[{",".join(feature_combo)}] vs {target_column}')
    plt.tight_layout()
    plt.show()
