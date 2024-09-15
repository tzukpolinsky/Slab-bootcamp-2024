
def interact_plot(data_tbl: pd.DataFrame, first_cat: str, second_cat: str, depended_vars: list,
                                       aggregation_function):
    # Get the unique levels of the categorical variables
    plot_data_tbl = data_tbl.copy()
    if data_tbl[first_cat].nunique() < data_tbl[second_cat].nunique():
        first_cat, second_cat = second_cat, first_cat
    for cat in [first_cat, second_cat]:
        if len(plot_data_tbl[cat].unique()) > 2:
            continue
        if cat == 'gender':
            plot_data_tbl[cat] = plot_data_tbl[cat].map({0: 'female', 1: 'male'})
        else:
            plot_data_tbl[cat] = plot_data_tbl[cat].astype(bool)
    first_cat_levels = plot_data_tbl[first_cat].unique()
    second_cat_levels = plot_data_tbl[second_cat].unique()

    # Generate markers and colors dynamically based on the unique levels
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'H', '*', 'x', 'input', '+',
               'd']  # A list of markers to choose from
    colors = plt.cm.rainbow(np.linspace(0, 1, len(second_cat_levels)))  # Generate a list of colors

    # Create subplots
    fig, axes = plt.subplots(1, len(depended_vars), figsize=(15, 5))

    for i, depended_var in enumerate(depended_vars):
        interaction_plot(
            plot_data_tbl[first_cat].to_numpy(),
            plot_data_tbl[second_cat].to_numpy(),
            plot_data_tbl[depended_var].to_numpy(),
            xlabel=first_cat,
            ylabel=second_cat,
            markers=[markers[j % len(markers)] for j in range(len(second_cat_levels))],  # Ensure we have enough markers
            colors=colors,  # Use the dynamically generated colors
            ms=10,
            ax=axes[i],
            func=aggregation_function
        )
        axes[i].set_title(f'Interaction Plot for {depended_var}')

    plt.tight_layout()
    plt.show()


palettes = {
    'type': {'E': '#BDD997', 'C': '#984788'},
    'substances_main_group': {
        "No Use": "#5765ab", "Light": "#50C878", "MDMA": "#F43183", "Stim": "#FF7F50", "Haluc": "#9966CC",
        "Other": '#DAA520'
    },
    'questionnaire': {'followup 1': 'lightgray', 'followup 2': 'lightgray'},
    'Got back to work since TE': {
        'didnt answered': '#e6194B',  # Bright Red
        'yes': '#3cb44b',  # Moderate Green
        'partially': '#ffe119',  # Bright outputellow
        'no': '#4363d8',  # Strong Blue
        'didnt worked before the event': '#f58231',  # Vivid Orange
        'yes, in the 3 month after TE': '#911eb4',  # Dark Violet
        'yes, in the last month': '#42d4f4',  # Sky Blue for yes in the last month
        'other': '#fabed4'  # Light Pink for other
    },
    'Got back to work since TE summarized': {
        'didnt answered': '#ff6347',  # Tomato Red
        'yes': '#2E8B57',  # Sea Green
        'partially': '#ffeb3b',  # Bright outputellow
        'no': '#DC143C'  # Crimson
    },
    'state change': {
        'yes -> yes': '#2E8B57',
        'no -> yes': '#2E8B57',
        'no -> no': '#DC143C',
        'yes -> no': '#DC143C',
    }
}
orders = {
    'Got back to work since TE summarized': {
        'didnt answered': 3,
        'yes': 0,
        'partially': 1,
        'no': 2
    },
    'substances_main_group': {
        "No Use": 0, "Light": 1, "MDMA": 4, "Stim": 3, "Haluc": 5,
        "Other": 6
    },
    'state change': {
        'yes -> yes': 0,
        'no -> yes': 1,
        'no -> no': 3,
        'yes -> no': 2,
    }
}


def hist_kde_plot(data_tbl: pd.DataFrame, column: str, bins: int = 10, title: str = None, xlabel: str = None,
                            ylabel: str = 'Frequency') -> list:
    """
    Plot a histogram with a Kernel Density Estimate (KDE) for a specified column in a pandas DataFrame.

    Parameters:
    data_tbl (pd.DataFrame): The input DataFrame containing the data.
    column (str): The name of the column to plot the histogram with KDE for.
    bins (int, optional): Number of bins for the histogram. Default is 10.
    title (str, optional): The title of the histogram. Default is None.
    xlabel (str, optional): The label for the x-axis. Default is None.
    ylabel (str, optional): The label for the y-axis. Default is 'Frequency'.

    Returns:
    a list of the bins of the plot

    Raises:
    KeyError: If the specified column does not exist in the DataFrame.
    TypeError: If the input DataFrame is not a pandas DataFrame.

    Example:
    >>> data_tbl = pd.DataFrame({'values': [1, 2, 2, 3, 4, 4, 4, 5]})
    >>> hist_kde_plot(data_tbl, 'values', bins=5, title='Histogram with KDE', xlabel='Values')
    """
    if not isinstance(data_tbl, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")

    if column not in data_tbl.columns:
        raise KeyError(f"The column '{column}' does not exist in the DataFrame.")

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data_tbl[column].dropna(), bins=bins, kde=True)
    if title:
        plt.title(title)
    else:
        plt.title(f'histogram with kde for {column},N:{len(data_tbl[column].dropna())}')
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    bar_patches = ax.patches
    bin_edges = [patch.get_x() for patch in bar_patches]
    return bin_edges


def group_hist_plot(data_tbl: pd.DataFrame, columns: list, bins: int = 10, title: str = None, xlabel: str = None,
                           ylabel: str = 'Frequency') -> np.array:
    """
    Plot grouped histograms for specified columns in a pandas DataFrame.

    Parameters:
    --------
    - data_tbl (pd.DataFrame): The input DataFrame containing the data.
    - columns (list): The list of columns to plot grouped histograms for.
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


def lasso_multi_plot(data_tbl: pd.DataFrame, in_features: [str], out_features: [str]):
    input = data_tbl[in_features]
    output = data_tbl[out_features]
    model = MultiTaskLassoCV().fit(input, output)
    output_pred = model.predict(input)
    data = pd.DataFrame(input, columns=in_features)
    data[out_features] = output
    data[[f'{col}_pred' for col in out_features]] = output_pred

    sns.pairplot(data)
    plt.suptitle('Pairplot of Predictors and Responses')
    plt.show()


def linear_plot(data_tbl: pd.DataFrame, y: np.array, feature_combo: [str], target_column: str, is_linear: bool,
                        p_value: float, fstat: float):
    """
    Plot the results of the linearity test.
    """
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
