import numpy as np
import pandas as pd
def group_hist_plot(data_tbl: pd.DataFrame, columns: list, bins: int = 10, title: str = None, xlabel: str = None,
                           ylabel: str = 'Frequency') -> np.array:

    """
    Plot grouped histograms for specified columns in a pandas DataFrame.

    Parameters:
    --------
    - data_tbl (pd.DataFrame): The input DataFrame containing the data.
    - columns (list): The list of columns to cat_count grouped histograms for.
    - bins (int, optional): Number of bins for the histogram. Default is 10.
    - title (str, optional): The title of the histogram. Default is None.
    - xlabel (str, optional): The label for the x-axis. Default is None.
    - ylabel (str, optional): The label for the y-axis. Default is 'Frequency'.

    Returns:
    --------
    the bins of the histogram
    """
    if not isinstance(data_tbl, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")

    for column in columns:
        if column not in data_tbl.columns:
            raise KeyError(f"The column '{column}' does not exist in the DataFrame.")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(data_tbl[columns].dropna(), bins=bins, alpha=0.5, edgecolor='black', label=columns,
                                stacked=True)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()
    return bins