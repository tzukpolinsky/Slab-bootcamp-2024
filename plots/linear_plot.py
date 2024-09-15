import numpy as np

import pandas as pd

def linear_plot(data_tbl: pd.DataFrame, y: np.array, feature_combo: [str], target_column: str, is_linear: bool,
                        p_value: float, fstat: float):
    """
    Plot the results of the linearity test.
    """
    import matplotlib.pyplot as plt
    n_features = data_tbl.shape[1]
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5))

    if n_features == 1:
        axes = [axes]  # Make axes iterable when there's only one feature
    for i, column in enumerate(data_tbl.columns):
        axes[i].scatter(data_tbl[column], y)
        axes[i].set_title(f'{column} vs {target_column}')
        axes[i].set_xlabel(column)
        if i == 0:
            axes[i].set_ylabel(f'{target_column}')
    plt.suptitle(
        f'Linearity Test Results (F={fstat:.2f}, p={p_value:.4f}) are: {"linear" if is_linear else "not linear"}')
    plt.title(f'[{",".join(feature_combo)}] vs {target_column}')
    plt.tight_layout()
    plt.show()
