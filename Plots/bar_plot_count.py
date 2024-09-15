from c import orders, palettes
import matplotlib.pyplot as plt


def cat_count_plot(data_tbl, category_column, subplot_column=None, title="", x_label=""):
    """
    Counts the number of items in each category of the specified column in the DataFrame
    and plots the counts as a bar chart. If subplot_column is provided, creates a subplot
    for each unique value in the subplot_column.

    Parameters:
    data_tbl (pd.DataFrame): The DataFrame containing the data.
    category_column (str): The name of the category column to count items in.
    subplot_column (str, optional): The name of the column to create subplots for. Default is None.

    Returns:
    dict: A dictionary with the counts of each category for each subplot (if subplot_column is provided).
    """
    plot_data_tbl = data_tbl.copy()
    if category_column not in plot_data_tbl.columns:
        raise ValueError(f"Column '{category_column}' does not exist in the DataFrame.")

    if subplot_column and subplot_column not in plot_data_tbl.columns:
        raise ValueError(f"Column '{subplot_column}' does not exist in the DataFrame.")
    palette = palettes[category_column] if category_column in palettes else None
    order = orders[category_column] if category_column in orders else None

    if subplot_column:
        unique_values = plot_data_tbl[subplot_column].unique()
        num_subplots = len(unique_values)
        fig, axes = plt.subplots(1, num_subplots, figsize=(8 * num_subplots, 6))
        counts_dict = {}

        for i, value in enumerate(unique_values):
            ax = axes[i] if num_subplots > 1 else axes
            ax.set_facecolor("white")
            subset = plot_data_tbl[plot_data_tbl[subplot_column] == value]
            counts = subset[category_column].value_counts()
            if order:
                index_order = sorted(counts.index.to_list(), key=lambda x: order[x])
                counts = counts.reindex(index_order)
            colors = ['#FF5733'] * len(counts)
            if palette is not None:
                for j, col in enumerate(counts.to_dict().keys()):
                    colors[j] = palette[col]
            total = counts.sum()
            counts.plot(kind='bar', ax=ax, color=colors)
            for p in ax.patches:
                height = p.get_height()
                percentage = 100 * height / total
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    height,
                    f'{percentage:.1f}%\nN:{height}',
                    ha='center',
                    va='bottom'
                )
            ax.set_suptitle(title)
            if x_label:
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel(category_column)
            ax.set_ylabel('')
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='center')
            ax.set_yticks([])
            counts_dict[value] = counts
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.show()
        return counts_dict

    else:
        category_counts = plot_data_tbl[category_column].value_counts()
        if order:
            index_order = sorted(category_counts.index.to_list(), key=lambda x: order[x])
            category_counts = category_counts.reindex(index_order)
        plt.figure(figsize=(10, 6))
        colors = ['#FF5733'] * len(category_counts)
        if palette is not None:
            for i, col in enumerate(category_counts.to_dict().keys()):
                colors[i] = palette[col]
        category_counts.plot(kind='bar', color=colors)
        total = category_counts.sum()
        ax = plt.gca()
        ax.set_facecolor("white")
        for p in plt.gca().patches:
            height = p.get_height()
            percentage = 100 * height / total
            ax.text(
                p.get_x() + p.get_width() / 2,
                height,
                f'{percentage:.1f}%\nN:{height}',
                ha='center',
                va='bottom'
            )
        plt.suptitle(title)
        if x_label:
            ax.set_xlabel(x_label)
        else:
            ax.set_xlabel(category_column)
        plt.ylabel('')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(rotation=30, ha='center')
        plt.yticks([])
        plt.show()

        return category_counts
