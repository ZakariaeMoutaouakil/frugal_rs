from typing import Dict

import h5py
import numpy as np


def analyze_hdf5_dataset(file_path: str) -> Dict[str, int]:
    """
    Analyzes the HDF5 file to determine the last example with non-zero predictions
    and the minimum inference count up to that point.

    Args:
    file_path (str): Path to the HDF5 file.

    Returns:
    dict: A dictionary containing:
        - 'last_nonzero_index': Index of the last example with non-zero predictions.
        - 'min_inference_count': Minimum inference count up to the last non-zero index.
    """
    with h5py.File(file_path, 'r') as f:
        # Assume the counts dataset name follows the pattern: "{dataset}_{sigma}_counts"
        # We'll find the first dataset that ends with "_counts"
        counts_dataset = next(name for name in f.keys() if name.endswith("_counts"))
        counts = f[counts_dataset][:]

    # Find the last non-zero count
    nonzero_indices = np.nonzero(counts)[0]

    if len(nonzero_indices) == 0:
        return {
            'last_nonzero_index': -1,
            'min_inference_count': 0
        }

    last_nonzero_index = nonzero_indices[-1]

    # Get the minimum count up to the last non-zero index
    min_inference_count = np.min(counts[:last_nonzero_index + 1][counts[:last_nonzero_index + 1] > 0])

    return {
        'last_nonzero_index': last_nonzero_index,
        'min_inference_count': min_inference_count
    }

# Example usage:
# result = analyze_hdf5_dataset('/path/to/your/hdf5/file.h5')
# print(result)
