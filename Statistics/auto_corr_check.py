import numpy as np
from typing import Tuple

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson


def auto_corr_check(leftovers: np.ndarray, alpha=0.05, with_conclusion_print=False,
                    print_autocorrelation_conclusion=None) -> Tuple[bool, float, float]:
    """
    Check for autocorrelation using the Durbin-Watson test and Ljung-Box test.

    Args:
        leftovers (np.ndarray): The leftovers from a linear regression model.
        alpha (float): The significance level for the test.
        with_conclusion_print (bool): print the conclusion of the test.

    Returns:
        Tuple[bool, float, float]: A tuple containing:
            - bool: True if no significant autocorrelation (Ljung-Box p-value > alpha), False otherwise.
            - float: The p-value of the Ljung-Box test.
            - float: The Durbin-Watson statistic.

    References:
    Durbin, J., & Watson, G. S. (1951). Testing for serial correlation in least
    squares regression. II. Biometrika, 38(1/2), 159-177.
    https://www.jstor.org/stable/2332325

    Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models.
    Biometrika, 65(2), 297-303.
    https://doi.org/10.1093/biomet/65.2.297
    :param with_conclusion_print:
    """
    dw_statistic = durbin_watson(leftovers)
    lb_test = acorr_ljungbox(leftovers, lags=[1], return_data_tbl=True)
    lb_p_value = lb_test['lb_pvalue'].values[0]
    if with_conclusion_print:
        print_autocorrelation_conclusion(lb_p_value > alpha, lb_p_value, dw_statistic, alpha)
    return lb_p_value > alpha, lb_p_value, dw_statistic
