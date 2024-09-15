

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