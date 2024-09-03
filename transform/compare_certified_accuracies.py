import pandas as pd


def calculate_certified_accuracy(df, radii):
    """
    Calculate certified accuracy at each radius for the dataset.

    Args:
    - df (DataFrame): The dataset DataFrame.
    - radii (list): List of radii values to consider.

    Returns:
    - accuracies (list): List of certified accuracies corresponding to each radius.
    """
    accuracies = []
    for r in radii:
        df_correct = df[(df['correct'] == 1) & (df['radius'] >= r)]
        certified_accuracy = len(df_correct) / len(df)
        accuracies.append(certified_accuracy)
    return accuracies


def compare_certified_accuracies(file_tuples, radii, labels):
    """
    Compare certified accuracies of two datasets at given radii.

    Args:
    - file_tuples (tuple of tuples): Each inner tuple contains two file paths for DataFrame loading.
    - radii (tuple): Tuple of radii values to consider.
    - labels (tuple): Tuple of labels for each dataset pair.

    Returns:
    - result_df (DataFrame): DataFrame with certified accuracies and comparisons.
    """
    result_data = []

    for (file1, file2), label in zip(file_tuples, labels):
        # Load DataFrames
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Calculate certified accuracies
        accuracies1 = calculate_certified_accuracy(df1, radii)
        accuracies2 = calculate_certified_accuracy(df2, radii)

        # Add results for the first dataset
        result_data.append([f"CS + Bonferroni - {label}"] + accuracies1)

        # Add results for the second dataset
        result_data.append([f"CS - {label}"] + accuracies2)

        # Calculate and add comparison row
        comparison = [f"{label} - Comparison (%)"] + [
            f"{(acc2 - acc1) / acc1 * 100:.2f}%" if acc1 != 0 else "N/A"
            for acc1, acc2 in zip(accuracies1, accuracies2)
        ]
        result_data.append(comparison)

    # Create DataFrame
    columns = ["Dataset"] + [f"Radius {r}" for r in radii]
    result_df = pd.DataFrame(result_data, columns=columns)

    return result_df


def main():
    # Set the options to display all rows and columns (None means no limit)
    pd.set_option('display.max_rows', None)  # for rows
    pd.set_option('display.max_columns', None)  # for columns
    methods = ('sequence_bonferroni', 'sequence')
    num_samples_range_first = range(100, 600, 200)
    num_samples_range_second = (100, 500, 1000)

    file_tuples = tuple(
        tuple(
            f'/home/pc/PycharmProjects/test_results/transformed/cifar10_0.50_{num_samples}_1/cifar10_0.50_{method}_first.csv'
            for method in methods) for num_samples in num_samples_range_first
    )

    labels = tuple(f'{num_samples} samples' for num_samples in num_samples_range_first)

    radii = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    result_df = compare_certified_accuracies(file_tuples, radii, labels)

    print(result_df)
    # Convert to LaTeX
    latex_code = result_df.to_latex()
    print(latex_code)

def main2():
    # Set the options to display all rows and columns (None means no limit)
    pd.set_option('display.max_rows', None)  # for rows
    pd.set_option('display.max_columns', None)  # for columns
    methods = ('sequence_bonferroni', 'sequence')
    num_samples_range_first = range(100, 600, 200)
    num_samples_range_second = (100, 500, 1000)

    file_tuples = tuple(
        tuple(
            f'/home/pc/PycharmProjects/test_results/transformed/cifar10_0.50_{num_samples}_1/cifar10_0.50_{method}_second.csv'
            for method in methods) for num_samples in num_samples_range_first
    )

    labels = tuple(f'{num_samples} samples' for num_samples in num_samples_range_first)

    radii = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    result_df = compare_certified_accuracies(file_tuples, radii, labels)

    print(result_df)


def main3():
    # Set the options to display all rows and columns (None means no limit)
    pd.set_option('display.max_rows', None)  # for rows
    pd.set_option('display.max_columns', None)  # for columns
    methods = ('cp', 'dichotomy')
    df_folder_path = f'/home/pc/PycharmProjects/test_results/transformed_discrete/cifar10_0.12_100/'
    # file_tuples = (tuple(df_folder_path + f'cifar10_0.12_' + method + '_first.csv' for method in methods),
    #                tuple(df_folder_path + f'cifar10_0.12_' + method + '_second.csv' for method in methods))
    file_tuples =((df_folder_path + f'cifar10_0.12_' + method + '_first.csv' for method in methods),)
    # print(file_tuples)
    # labels = ('First', 'Second')
    labels = ('',)

    # radii = ( 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5)
    radii = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    result_df = compare_certified_accuracies(file_tuples, radii, labels)

    print(result_df)


if __name__ == "__main__":
    main3()
