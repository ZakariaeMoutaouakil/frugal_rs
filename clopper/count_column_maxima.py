from torch import cuda, Tensor, argmax, bincount, tensor


def count_column_maxima(matrix: Tensor) -> Tensor:
    """
    Count how many times each column has the maximum element in a row.

    Args:
    matrix (torch.Tensor): A 2D PyTorch tensor

    Returns:
    torch.Tensor: A 1D tensor where the element at each index is the count of
                  how many times that column has the maximum element in a row
    """
    # Get the indices of the maximum element in each row
    max_indices = argmax(matrix, dim=1)

    # Count the occurrences of each column index
    column_counts = bincount(max_indices, minlength=matrix.shape[1])

    # Ensure the output is on the same device as the input matrix
    return column_counts.to(matrix.device)


def main():
    # Example usage
    # Create a 2D tensor on CPU
    matrix_cpu = tensor([
        [1, 5, 3],
        [2, 2, 4],
        [3, 1, 3],
        [4, 4, 4]
    ])

    print("Input matrix (CPU):")
    print(matrix_cpu)

    result_cpu = count_column_maxima(matrix_cpu)
    print("\nColumn maxima counts (CPU):")
    print(result_cpu)

    # If CUDA is available, demonstrate with a GPU tensor
    if cuda.is_available():
        matrix_gpu = matrix_cpu.cuda()
        print("\nInput matrix (GPU):")
        print(matrix_gpu)

        result_gpu = count_column_maxima(matrix_gpu)
        print("\nColumn maxima counts (GPU):")
        print(result_gpu)
        print("Device of result:", result_gpu.device)

    # Explanation of the result
    for i, count in enumerate(result_cpu):
        print(f"Column {i} has the maximum {count.item()} time(s)")


if __name__ == "__main__":
    main()
