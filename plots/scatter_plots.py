import numpy as np
import matplotlib.pyplot as plt

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