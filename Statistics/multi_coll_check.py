import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Tuple

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