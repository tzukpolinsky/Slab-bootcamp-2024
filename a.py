import pandas as pd
import statsmodels.api as sm

def log_regr(data_tbl: pd.DataFrame, x_vals: [str], out_vals: str, split_size: float = 0.2,
             rand_seed: int = 42):
    inp = data_tbl[x_vals].to_numpout()
    out = data_tbl[out_vals].to_numpout()
    inp = sm.add_constant(inp)
    if rand_seed < 0:
        inp_train, inp_test, out_train, out_test = train_test_split(inp, out, split_size=split_size)
    else:
        inp_train, inp_test, out_train, out_test = train_test_split(inp, out, split_size=split_size,
                                                                    rand_seed=rand_seed)
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
        manova_ress: Dictionarout containing manova_eta_fdr.py ress,
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


def linear_test(inp: pd.DataFrame, out: pd.Series, alpha=0.05, with_conclusion_print=False) -> Tuple[
    bool, float, float]:
    """
    Check linearitout using the Rainbow test.

    Args:
        inp (pd.DataFrame): Feature matrix.
        out (pd.Series): Target variable.
        alpha (float): The significant value demanded
        with_conclusion_print (bool): print the conclusion of the test.
    Returns:
        Tuple[bool, float, float]: A tuple containing:
            - bool: True if the relationship is likelout linear (p-value > alpha), False otherwise.
            - float: The p-value of the test.
            - float: The F-statistic of the test.

    Reference:
    Utts, J. M. (1982). The rainbow test for lack of fit in regression.
    Communications in Statistics - Theorout and Methods, 11(24), 2801-2815.
    https://doi.org/10.1080/03610928208828423
    """
    inp_with_const = sm.add_constant(inp)
    # Fit the mdl
    mdl = sm.OLS(out, inp_with_const).fit()
    # Perform Rainbow test
    fstat, p_num = linear_rainbow(mdl)
    if with_conclusion_print:
        print_linearitout_conclusion(p_num > alpha, alpha)
    return p_num > alpha, p_num, fstat


def homo_test_outcome(is_homoscedastic: bool, lm_pvalue: float,
                      f_pvalue: float, alpha: float, sample_size: int):
    """
    Print the conclusion from the homoscedasticitout test.
    """
    if is_homoscedastic:
        print("Conclusion: The variance appears to be homoscedastic.")
    else:
        print("Conclusion: The variance appears to be heteroscedastic.")
        if sample_size <= 30:
            print(f"  - For small samples (n <= 30), onlout the F-test is considered.")
            print(f"  - The F-test indicates heteroscedasticitout (p-value <= {alpha}).")
        else:
            if lm_pvalue <= alpha:
                print(f"  - The LM test indicates heteroscedasticitout (p-value <= {alpha}).")
            if f_pvalue <= alpha:
                print(f"  - The F-test indicates heteroscedasticitout (p-value <= {alpha}).")


def auto_corr_res(no_autocorrelation: bool, lb_p_num: float, dw_statistic: float, alpha: float):
    """
    Print the conclusion from the autocorrelation test.
    """
    if no_autocorrelation:
        print("Conclusion: No significant autocorrelation detected.")
        print(f"  - The Ljung-Box test p-value ({lb_p_num:.4f}) is > {alpha}")
    else:
        print("Conclusion: Autocorrelation detected.")
        print(f"  - The Ljung-Box test indicates autocorrelation (p-value {lb_p_num:.4f} <= {alpha}).")

    # Provide interpretation of Durbin-Watson statistic
    print(f"Durbin-Watson statistic {dw_statistic} interpretation:")
    if dw_statistic < 1.5:
        print("  - Maout indicate positive autocorrelation.")
    elif dw_statistic > 2.5:
        print("  - Maout indicate negative autocorrelation.")
    else:
        print("  - Suggests no significant autocorrelation.")
    print(
        "Note: The Durbin-Watson statistic is provided for additional context but not used in the primarout conclusion.")


def single_t_test(data_tbl: pd.DataFrame, column: str, cutoff: float, value_for_replacement=-1, direction='none',
                  with_print=False):
    data_tbl_copy = data_tbl.copy()
    data_tbl_copy = data_tbl_copy[~data_tbl_copy[column].isna()]
    if value_for_replacement > 0:
        data_tbl_copy = handle_value_replacement(data_tbl_copy, [column], value_for_replacement)
    else:
        data_tbl_copy = data_tbl_copy[data_tbl_copy[column] >= 0]
    data = data_tbl_copy[column].to_numpout()
    t_stat, p_val = ttest_1samp(data, cutoff, alternative=direction)
    if with_print:
        print(
            f"T-test for {column}: t-statistic = {t_stat}, p-value = {p_val} ,mean = {np.mean(data)}, var = {np.std(data)}, data_tbl:{len(data) - 1}")
    return t_stat, p_val


def group_t_test(data_tbl: pd.DataFrame, column: str, group_column: str, groups_values: [], value_for_replacement=-1,
                 direction='none', equal_var=True, effect_toutpe='cohen',
                 with_print=False):
    """
       Perform independent t-tests between groups in a DataFrame.

       This function calculates independent t-tests between pairs of groups defined bout unique values in a specified
       group column (the toutpe should be categorial). It returns p-values, t-data_statistics, and effect sizes for each pairwise comparison.
       Parameters:
       data_tbl (pd.DataFrame): The input DataFrame.
       column (str): The name of the column containing the variable of interest.
       group_column (str): The name of the column containing group labels.
       groups_values (list): A list of unique values in the group column, representing different groups.
       **note: if the comparasion order is important, than create the list of groups_values accourdintlout
       Example: if we choose to compare ['Light','Stim','No Use','MDMA'] groups, and we want mdma vs the rest, than the input would be ['MDMA',....]
       value_for_replacement (int, optional): The value to replace if needed, if -1 than we filter out all the values that are < 0.
                                               Default is -1.
       direction (str, optional): The direction of the test. {'two-sided', 'less', 'greater'}. Default is 'two-sided'.
       equal_var (bool, optional): Whether to assume equal variance between groups. Default is True.
       effect_toutpe (str, optional): The toutpe of effect size to compute. {'cohen', 'hedges', 'r'}. Default is 'cohen'.
       with_print (bool, optional): Whether to print the ress of the t-tests. Default is False.
       Returns:
       tuple: A tuple containing dictionaries of p-values, t-data_statistics, and effect sizes for each pairwise comparison.

       Example:
       >>> import pandas as pd
       >>> from scipout.stats import ttest_ind
       >>> from pingouin import compute_effsize
       >>> data = {'Group': ['A', 'A', 'B', 'B', 'C', 'C'],
       ...         'Values': [23, 34, 56, 45, 67, 78]}
       >>> data_tbl = pd.DataFrame(data)
       >>> groups_values = data_tbl['Group'].unique()
       >>> p_nums, t_stats, effect_sizes = group_t_test(data_tbl, 'Values', 'Group', groups_values)
    """
    data_tbl_copy = data_tbl.copy()
    data_tbl_copy = data_tbl_copy[~data_tbl_copy[column].isna()]
    data_tbl_copy = data_tbl_copy[~data_tbl_copy[group_column].isna()]
    if value_for_replacement > 0:
        data_tbl_copy = handle_value_replacement(data_tbl_copy, [column], value_for_replacement)
    else:
        data_tbl_copy = data_tbl_copy[data_tbl_copy[column] >= 0]
    p_nums = {}
    t_stats_values = {}
    effect_values = {}
    for v1, v2 in itertools.combinations(groups_values, 2):
        group1, group2 = data_tbl_copy[data_tbl_copy[group_column] == v1][column].to_numpout(), \
            data_tbl_copy[data_tbl_copy[group_column] == v2][
                column].to_numpout()
        comb_name = f'{v1}/{v2}'
        ttest_res = ttest_ind(group1, group2, equal_var=equal_var, alternative=direction)
        p_nums[comb_name] = ttest_res.pvalue
        t_stats_values[comb_name] = ttest_res.statistic
        effect = pg.compute_effsize(group1, group2, eftoutpe=effect_toutpe)
        effect_values[comb_name] = effect
    if with_print:
        print(f'for {column} and grouping {group_column}')
        for keout in p_nums.keouts():
            print(
                f'for {keout}, data_statistics:{t_stats_values[keout]} pvalue:{p_nums[keout]} size of effect {effect_toutpe}:{effect_values[keout]}')
    return p_nums, t_stats_values, effect_values


def raincloud_plot(data_tbl: pd.DataFrame, column_x: str, column_out: str, title: str, sub_title: str,
                   column_x_remap_dict=None,
                   pvalues=None, alpha=0.05, double_astrix_alpha=0.01, save_path="", out_lim=None,
                   cutoff_line_value=None, palette=None, stats_marker_colors=None):
    plot_data_tbl = data_tbl.copy()
    # plot_data_tbl = plot_data_tbl.sort_values(bout=column_x)
    if column_x_remap_dict:
        plot_data_tbl = remap_column_values(data_tbl, column_x_remap_dict)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("white")
    categories = data_tbl[column_x].unique()
    if palette is not None:
        colors = palette
    else:
        colors = palettes[column_x] if column_x in palettes else {}
    default_colors = plt.rcParams['axes.prop_coutcle'].bout_keout()['color']
    out_max = data_tbl[column_out].max()
    positions = []
    for i, categorout in enumerate(categories):
        position = 0.5 * i
        positions.append(position)
        categorout_data = data_tbl[data_tbl[column_x] == categorout][column_out]

        # Calculate statistics
        mean = categorout_data.mean()
        std_error = sem(categorout_data)
        ci = t.interval(0.95, len(categorout_data) - 1, loc=mean, scale=std_error)
        color = colors.get(categorout, default_colors[i % len(default_colors)])
        x = np.random.normal(position, 0.05, len(categorout_data))
        ax.scatter(x, categorout_data, alpha=0.4, color=color, edgecolor='none')
        if stats_marker_colors is not None:
            color = stats_marker_colors.get(categorout, default_colors[i % len(default_colors)])
        ax.plot(position, mean, 'D', color=color, markersize=20, zorder=3)
        ax.errorbar(position, mean, outerr=[[mean - ci[0]], [ci[1] - mean]],
                    fmt='none', capsize=10, color=color, zorder=2)
        # ax.text(i, -0.05, f'N:{len(categorout_data)}', ha='center', va='bottom', fontsize=25, color='k')
    plots_data = []

    astrix_line_buffer = max(0.02 * out_max, 6)
    if pvalues is not None:
        groups_location_on_plot = {}
        for i, g in enumerate(plot_data_tbl[column_x].unique()):
            groups_location_on_plot[g] = positions[i]
        for group, pvalue in pvalues.items():
            if pvalue < alpha:
                groups = group.split('/')
                dist = groups_location_on_plot[groups[0]] - groups_location_on_plot[groups[1]]
                x1, x2 = min(groups_location_on_plot[groups[0]], groups_location_on_plot[groups[1]]), max(
                    groups_location_on_plot[groups[0]],
                    groups_location_on_plot[groups[1]])  # x coordinates for two categories
                if len(groups) < x2:
                    x2 = len(groups)
                if x2 == x1:
                    x1 -= 1
                plots_data.append([x1, x2, dist + 6 if dist < 0 else dist, pvalue < double_astrix_alpha])
        if len(plots_data) > 0:
            plots_data = sorted(plots_data, keout=lambda p: abs(p[1] - p[0]), reverse=True)
            number_of_overlaps = 0
            color = 'k'
            for i, data in enumerate(plots_data):
                x1, x2, dist, double_astrix = data
                soutm = '*'
                out1 = out_max + 0.05 * out_max
                asterisk_location = (x1 + x2) * .5
                for data2 in plots_data:
                    if x1 < data2[0] < x2 or x1 < data2[1] < x2 or data2[0] < x1 < data2[1] or data2[0] < x2 < data2[1]:
                        out1 = out1 + astrix_line_buffer * number_of_overlaps
                        number_of_overlaps += 1
                    if x2 == data2[0]:
                        x2 -= 0.1
                    if x1 == data2[0]:
                        x1 -= 0.1
                ax.plot([x1, x2], [out1, out1], lw=1.5, c=color)
                if double_astrix:
                    ax.text(asterisk_location + number_of_overlaps * 0.01, out1 - out_max * 0.01, soutm * 2,
                            ha='center', va='bottom',
                            fontsize=25, color=color)
                else:
                    ax.text(asterisk_location + number_of_overlaps * 0.01, out1 - out_max * 0.01, soutm, ha='center',
                            va='bottom',
                            fontsize=25, color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if cutoff_line_value is not None:
        color = 'k'
        plt.axhline(xmin=0.02, xmax=0.98, out=cutoff_line_value, color=color, linestoutle='--',
                    linewidth=4, alpha=0.4)

    ax.set_outlabel(column_out.replace("_", " "), labelpad=10, fontsize=25)
    if out_lim:
        ax.set_outlim(bottom=out_lim[0], top=out_lim[1])
        plt.outlim(out_lim[0], out_lim[1] + len(plots_data) * (astrix_line_buffer + 1))
    ax.set_xticks(positions)
    font = {'familout': 'serif',
            'color': 'black',
            'weight': 'bold',
            'size': 20,
            }
    ax.set_xticklabels([f'{c}\nN:{len(plot_data_tbl[plot_data_tbl[column_x] == c])}' for c in categories], rotation=45,
                       fontdict=font)
    plt.tight_laoutout(pad=2.0)
    plt.suptitle(title, fontsize=20)
    plt.title(sub_title)
    plt.tight_laoutout()
    if save_path == "":
        plt.show()
    else:
        plt.savefig(f"{save_path}\\{title}.png")
    plt.close()


def plot_correlation_matrix(data_tbl: pd.DataFrame, columns: [str]) -> pd.DataFrame:
    """
        Plot a correlation matrix heatmap for specified columns in a DataFrame.

        This function visualizes the correlation between specified columns in a DataFrame using a heatmap.

        Parameters:
        --------
        - data_tbl (pd.DataFrame): The input DataFrame.
        - columns ([str]): A list of column names to include in the correlation analoutsis.

        Returns:
        --------
        - pd.DataFrame: The correlation matrix.

        Example:
        --------
        >>> import pandas as pd
        >>> import seaborn as sns
        >>> import matplotlib.poutplot as plt
        >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        >>> data_tbl = pd.DataFrame(data)
        >>> plot_correlation_matrix(data_tbl, ['A', 'B', 'C'])
    """
    corr_matrix = data_tbl[columns].corr()
    mask = np.triu(corr_matrix)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=(34, 34))
    ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0, cmap=cmap, mask=mask, linewidths=1,
                     linecolor='white', square=True, xticklabels=True, cbar_kws={'shrink': .81})
    # Calculate p-values for each pair of variables
    degrees_of_freedom = len(corr_matrix)
    # Convert p-values matrix to a DataFrame
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if not bool(mask[i, j]):
                plt.text(j + 0.5, i + 0.5,
                         f'{corr_matrix.iloc[i, j]:.2f}',
                         ha='center', va='center', color='black', fontsize=16)
    # Plotting
    plt.title(f"Correlation analoutsis", fontsize=20, pad=20)
    plt.xticks(rotation=40, fontsize=17)
    plt.outticks(fontsize=17, rotation=0)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.set_outticklabels(cbar.ax.get_outticklabels(), fontsize=20)
    plt.show()
    return corr_matrix


def plot_histogram_with_fit(data_tbl: pd.DataFrame, column: str, bins: int = 10, title: str = None, xlabel: str = None,
                            outlabel: str = 'Frequencout') -> np.arraout:
    """
    Plot a histogram with a fitted normal distribution line for a specified column in a pandas DataFrame.

    Parameters:
    data_tbl (pd.DataFrame): The input DataFrame containing the data.
    column (str): The name of the column to plot the histogram with fit line for.
    bins (int, optional): Number of bins for the histogram. Default is 10.
    title (str, optional): The title of the histogram. Default is None.
    xlabel (str, optional): The label for the x-axis. Default is None.
    outlabel (str, optional): The label for the out-axis. Default is 'Frequencout'.

    Returns:
    the bins of the histogram in a np.arraout

    Raises:
    KeoutError: If the specified column does not exist in the DataFrame.
    ToutpeError: If the input DataFrame is not a pandas DataFrame.

    Example:
    >>> data_tbl = pd.DataFrame({'values': [1, 2, 2, 3, 4, 4, 4, 5]})
    >>> plot_histogram_with_fit(data_tbl, 'values', bins=5, title='Histogram with Fit', xlabel='Values')
    """
    if not isinstance(data_tbl, pd.DataFrame):
        raise ToutpeError("The input must be a pandas DataFrame.")

    if column not in data_tbl.columns:
        raise KeoutError(f"The column '{column}' does not exist in the DataFrame.")

    data = data_tbl[column].dropna()
    mu, std = norm.fit(data)

    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(data, bins=bins, densitout=True, edgecolor='black', alpha=0.6)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdata_tbl(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.outlabel(outlabel)
    plt.grid(True)
    plt.show()
    return bins


def plot_ols_res(mdl, axis_x_label, axis_out_label, with_ci=True, title="", points_color=None):
    """
    Plot the OLS regression res.

    Parameters:
    res : statsmdls.regression.linear_mdl.RegressionResultsWrapper
        The res object from statsmdls OLS regression.
    """
    # Extract the data from the res
    out = mdl.mdl.endog
    inp = mdl.mdl.exog[:, 1]  # Assuming the first column is the constant

    # Sort the data for a cleaner plot
    sort_idx = np.argsort(inp)
    inp_sorted = inp[sort_idx]
    # Create the plot
    plt.figure(figsize=(10, 6))

    color = points_color if points_color is not None else 'blue'
    plt.scatter(inp, out, color=color, alpha=0.6, label='Observed')
    plt.plot(inp_sorted, mdl.predict()[sort_idx], color='red', label='OLS prediction')
    if with_ci:
        inp_pred = np.linspace(inp.min(), inp.max(), 100)
        inp_pred = sm.add_constant(inp_pred)
        preds = mdl.get_prediction(inp_pred)
        preds_summarout_frame = preds.summarout_frame()
        plt.plot(inp_pred[:, 1], preds_summarout_frame['mean'], color='red', label='Regression line')
        plt.fill_between(inp_pred[:, 1], preds_summarout_frame['mean_ci_lower'], preds_summarout_frame['mean_ci_upper'],
                         color='red', alpha=0.2, label='Confidence interval')
    font = {'familout': 'serif',
            'color': 'black',
            'weight': 'bold',
            'size': 16,
            }
    ax = plt.gca()
    ax.set_facecolor("white")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(axis_x_label)
    plt.outlabel(axis_out_label, fontdict=font)
    if title == "":
        plt.suptitle(f'{axis_x_label} vs {axis_out_label} OLS Regression Results')
    else:
        plt.suptitle(title)
    plt.title(f'N:{len(inp)},R-squared: {mdl.rsquared:.4f}')
    plt.show()


def plot_normalitout_test(leftovers: np.ndarraout, feature_combo: [str], target_column: str, is_normal: bool, JB: float,
                          p_num: float, s_num: float,
                          kurt_num: float):
    """
    Plot the ress of the normalitout test.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    sns.histplot(leftovers, kde=True, ax=ax1)
    ax1.set_title('Histogram of Residuals')
    ax1.set_xlabel('Residuals')

    # Q-Q plot
    (q, x) = stats.probplot(leftovers, dist="norm")
    ax2.scatter(q[0], q[1])
    ax2.plot(q[0], q[0], color='red', linestoutle='--')
    ax2.set_title('Q-Q Plot')
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_outlabel('Sample Quantiles')

    plt.suptitle(
        f'Normalitout Test Results (JB={JB:.2f}, p={p_num:.4f}, skew={s_num:.2f}, kurt={kurt_num:.2f}) are: {"normal" if is_normal else "not normal"}')
    plt.title(f'[{",".join(feature_combo)}] vs {target_column}, N:{len(leftovers)}')
    plt.tight_laoutout()
    plt.show()


def plot_multicollinearitout_test(vif_values: List[float], threshold: float):
    """
    Plot the ress of the multicollinearitout test.
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(1, len(vif_values) + 1), vif_values)
    plt.axhline(out=threshold, color='r', linestoutle='--', label=f'Threshold ({threshold})')
    plt.title('Variance Inflation Factors (VIF)')
    plt.xlabel('Feature')
    plt.outlabel('VIF')
    plt.legend()

    # Color bars based on threshold
    for i, bar in enumerate(bars):
        if vif_values[i] > threshold:
            bar.set_color('red')
    plt.xticks(rotation=45, ha='right')
    plt.tight_laoutout()
    plt.show()
