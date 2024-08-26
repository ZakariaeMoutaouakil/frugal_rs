import os
from argparse import ArgumentParser
from os.path import splitext
from time import time
from typing import List

from matplotlib.pyplot import figure, ylabel, plot, legend, grid, savefig, show, xlabel, title
from numpy import sort, append
from pandas import DataFrame, read_csv

parser = ArgumentParser(description='Transform dataset')
parser.add_argument("--dir", type=str, help="results directory", required=True)
parser.add_argument("--out", type=str, help="output directory", required=True)
args = parser.parse_args()


def process_file(df: DataFrame):
    """Load and process the dataset to calculate certified accuracy for various radii."""
    # Filter rows where the prediction is correct
    df_correct = df[df['correct'] == 1]  # Filtering the dataframe to include only correct predictions

    # Sort by radius
    df_correct = df_correct.sort_values(by='radius')  # Sorting the filtered dataframe by the 'radius' column

    # Unique radius values in the dataset
    unique_radii = sort(df_correct['radius'].unique())  # Extracting unique radius values from the sorted dataframe

    # Calculate the certified accuracy at each unique radius
    certified_accuracies = []  # List to store certified accuracies
    for r in unique_radii:
        certified_count = len(df_correct[df_correct['radius'] >= r])  # Counting certified predictions for each radius
        certified_accuracy = certified_count / len(df)  # Calculating certified accuracy
        certified_accuracies.append(certified_accuracy)  # Appending certified accuracy to the list

    # Append a final point projecting the curve to the x-axis
    if len(unique_radii) > 0:
        last_r = unique_radii[-1]  # Extracting the last radius value
        unique_radii = append(unique_radii, last_r + 0.01)  # Adding a small increment to project the curve
        certified_accuracies.append(0)  # Appending 0 to certified accuracies for the final point

    return unique_radii, certified_accuracies  # Returning unique radii and certified accuracies


def plot_multiple_files(dfs: List[DataFrame], labels: List[str], title_name: str = None, save_path: str = None) -> None:
    """Plot certified accuracy curves from multiple files in the same figure."""
    figure(figsize=(12, 8))  # Creating a new figure with specified size

    for i, (df, label) in enumerate(zip(dfs, labels)):
        unique_radii, certified_accuracies = process_file(df)  # Processing each file

        plot(unique_radii, certified_accuracies, linestyle='-', label=label)  # Plotting certified accuracy curve

    xlabel('Radius (r)')  # Setting the label for x-axis
    ylabel('Certified Accuracy')  # Setting the label for y-axis
    # Setting the title of the plot
    title('Certified Accuracy in Terms of Radius') if not title_name else title(title_name)
    legend()  # Displaying legend
    grid(False)  # Removing the grid lines from the plot

    # Save the plot if save_path is provided
    if save_path:
        savefig(save_path)  # Saving the plot to the specified path
    else:
        show()  # Displaying the plot


name_map = {
    'bernstein_bonferroni': 'Bernstein + Bonferroni',
    'bernstein': 'Bernstein',
    'sequence': 'CS',
    'sequence_bonferroni': 'CS + Bonferroni',
    'cp': 'CP + Bonferroni',
    'dichotomy': 'Dichotomy',
}


def main():
    log_file_name = splitext([f for f in os.listdir(args.dir) if f.endswith('.log')][0])[0]
    dataset_type = log_file_name.split('_')[0]
    dataset = 'Cifar10' if dataset_type == 'cifar10' else 'Imagenet'
    sigma = log_file_name.split('_')[1]

    first_title = f'Certified Accuracy in Terms of Radius for {dataset} with σ={sigma}'
    first_paths = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith('first.csv')]
    first_labels = []
    for file_path in first_paths:
        # Remove the extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Split into components
        parts = base_name.split('_')

        # Combine the first and last parts accordingly
        method = '_'.join(parts[2:-1])
        radius = parts[-1]

        if radius == 'second':
            continue

        first_labels.append(name_map[method])

    plot_multiple_files([read_csv(f, delimiter=',') for f in first_paths], first_labels, first_title,
                        os.path.join(args.dir, 'first.png'))

    second_title = f'Certified Accuracy in Terms of Radius for {dataset} with σ={sigma}'
    second_paths = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith('second.csv')]
    second_labels = []
    for file_path in second_paths:
        # Remove the extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Split into components
        parts = base_name.split('_')

        # Combine the first and last parts accordingly
        method = '_'.join(parts[2:-1])
        radius = parts[-1]

        if radius == 'first':
            continue

        second_labels.append(name_map[method])

    plot_multiple_files([read_csv(f, delimiter=',') for f in second_paths], second_labels, second_title,
                        os.path.join(args.dir, 'second.png'))


if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    print(f"Total time: {end_time - start_time:.4f}" + " seconds")
