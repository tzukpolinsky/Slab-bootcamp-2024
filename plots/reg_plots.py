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
