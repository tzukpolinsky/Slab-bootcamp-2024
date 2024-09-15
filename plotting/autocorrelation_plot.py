import numpy as np


def auto_corr_plot(residuals: np.ndarray, feature_combo:[str], target_column:str, no_autocorrelation: bool, lb_p_res: float, dw_statistic: float,
                   statsmdls=None, plot_acf=None):
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