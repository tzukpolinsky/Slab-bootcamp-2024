import pd
import sns
from matplotlib import pyplot as plt


def time_comp_plot(data_set: pd.DataFrame, category_col: str, value_col: str, time_col: str, title: str):
    """
    Create a line plot showing the change in values for each category between two time points.

    Parameters:
    data_set (pandas.DataFrame): The input DataFrame containing the data.
    category_col (str): The name of the column containing category labels (e.g., subject names).
    value_col (str): The name of the column containing the values to be plotted.
    title (str): The title for the plot.

    Returns:
    matplotlib.figure.Figure: The created figure object.

    The function expects the DataFrame to have the following structure:
    - A column for categories (e.g., subject names)
    - A column for time points (assumed to have two unique values)
    - A column for values

    Example usage:
    >>> import pandas as pd
    >>> data = {
    ...     'Subject': ['A', 'B', 'C', 'A', 'B', 'C'],
    ...     'Time': [1, 1, 1, 2, 2, 2],
    ...     'Value': [10, 15, 8, 12, 14, 10]
    ... }
    >>> data_set = pd.DataFrame(data)
    >>> time_comp_plot(data_set, 'Subject', 'Value', 'Change in Values')
    """
    plt.figure(figsize=(10, 6))
    plot_data_set = data_set.copy()
    plot_data_set = plot_data_set.sort_values(by=value_col)
    # Create the lineplot
    sns.lineplot(data=plot_data_set, x=time_col, y=value_col, hue=category_col, marker='o')

    # Customize the plot
    plt.title(title)
    plt.xlabel('Time Point')
    plt.ylabel(value_col)

    # Add value labels
    for line in plt.gca().lines:
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            plt.text(x, y, f' {y:.1f}', va='center', ha='left')
    plt.legend().remove()
    plt.tight_layout()
    plt.show()
