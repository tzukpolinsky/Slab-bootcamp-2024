import numpy as np
import pandas as pd


def interaction_plot(param, param1, param2, xlabel, ylabel, markers, colors, ms, ax, func):
    pass


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
    import matplotlib.pyplot as plt
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
                  ylabel: str = 'Frequency', sns=None) -> list:
    """
    Plot a histogram with a Kernel Density Estimate (KDE) for a specified column in a pandas DataFrame.

    Parameters:
    data_tbl (pd.DataFrame): The input DataFrame containing the data.
    column (str): The name of the column to cat_count the histogram with KDE for.
    bins (int, optional): Number of bins for the histogram. Default is 10.
    title (str, optional): The title of the histogram. Default is None.
    xlabel (str, optional): The label for the x-axis. Default is None.
    ylabel (str, optional): The label for the y-axis. Default is 'Frequency'.

    Returns:
    a list of the bins of the cat_count

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
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data_tbl[column].dropna(), bins=bins, kde=True)
    if title:
        import matplotlib.pyplot as plt
        plt.title(title)
    else:
        import matplotlib.pyplot as plt
        plt.title(f'histogram with kde for {column},N:{len(data_tbl[column].dropna())}')
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    bar_patches = ax.patches
    bin_edges = [patch.get_x() for patch in bar_patches]
    return bin_edges
