from time import time
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from plot import process_file, name_map


def plot_multiple_files(ax, dfs: Tuple[pd.DataFrame, ...], labels: Tuple[str, ...], title_name: str, y_label_name: str) \
        -> None:
    """Plot certified accuracy curves from multiple files on a given axis."""

    for df, label in tqdm(zip(dfs, labels), desc='Plotting axis', leave=False, total=len(dfs)):
        unique_radii, certified_accuracies = process_file(df)  # Processing each file

        ax.plot(unique_radii, certified_accuracies, linestyle='-', label=label)  # Plotting certified accuracy curve

    ax.set_xlabel('Radius (r)')  # Setting the label for x-axis
    if y_label_name:
        # Create a label with different font weights
        ax.set_ylabel(y_label_name)

        # Get the current label
        ylabel = ax.get_ylabel()

        # Split the label into lines
        lines = ylabel.split('\n')
        print(lines)

        # Create a new label with the first line bold
        new_label = f'$\\mathbf{{{lines[0]}}}$\n{lines[1]}'
        print(new_label)

        # Set the new label
        ax.set_ylabel(new_label)
    else:
        ax.set_ylabel('Certified Accuracy')
    # Setting the title of the plot
    ax.set_title(title_name, fontweight='bold')

    ax.legend()  # Displaying legend
    ax.grid(False)  # Removing the grid lines from the plot


def main():
    methods = ('bernstein_bonferroni', 'bernstein', 'sequence', 'sequence_bonferroni')
    # Number of plots
    num_samples_range = range(100, 600, 200)
    # Create figure and axes
    fig, axs = plt.subplots(2, len(num_samples_range), figsize=(15, 10))  # 1 row, n columns

    for i in tqdm(range(len(num_samples_range)), desc='Plotting'):
        num_samples = num_samples_range[i]
        title_name = f'{num_samples} samples'
        df_folder_path = f'/home/pc/PycharmProjects/test_results/transformed/cifar10_0.50_{num_samples}_1/'
        df_paths = tuple(df_folder_path + 'cifar10_0.50_' + method + '_first.csv' for method in methods)
        dfs = tuple(pd.read_csv(path) for path in df_paths)
        labels = tuple(name_map[method] for method in methods)

        ax = axs[0, i]
        y_label_name = 'First\\ Margin\n Certified Accuracy' if i == 0 else None
        plot_multiple_files(ax, dfs, labels, title_name, y_label_name)

        df_paths = tuple(
            df_folder_path + 'cifar10_0.50_' + method + '_first.csv' for method in methods if method != 'bernstein'
        )
        dfs = tuple(pd.read_csv(path) for path in df_paths)
        labels = tuple(name_map[method] for method in methods if method != 'bernstein')

        title_name = ''
        ax = axs[1, i]
        y_label_name = 'Second\\ Margin\n Certified Accuracy' if i == 0 else None
        plot_multiple_files(ax, dfs, labels, title_name, y_label_name)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


if __name__ == "__main__":
    start_time = time()
    main()
    print(f'Elapsed time: {time() - start_time:.4f} seconds')
