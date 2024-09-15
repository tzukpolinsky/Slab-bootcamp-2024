
import matplotlib.pyplot as plt

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