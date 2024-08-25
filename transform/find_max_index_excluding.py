from time import time

import torch


def find_max_index_excluding(tensor: torch.Tensor, i: int) -> int:
    # Create a copy of the tensor
    temp_tensor = tensor.clone()

    # Set the value at index i to the minimum possible value based on the tensor's dtype
    if torch.is_floating_point(tensor):
        temp_tensor[i] = torch.finfo(tensor.dtype).min
    else:
        temp_tensor[i] = torch.iinfo(tensor.dtype).min

    # Find the argmax of the modified tensor
    max_index = torch.argmax(temp_tensor)

    return max_index.item()


def main() -> None:
    # Example usage with integer tensor:
    tensor = torch.tensor([3, 7, 2, 9, 5, 1])
    i = 3  # Index to exclude (9 in this case)
    start_time = time()
    result = find_max_index_excluding(tensor, i)
    print(f"Time taken: {time() - start_time} seconds")
    print(f"Original tensor: {tensor}")
    print(f"Index to exclude: {i}")
    print(f"Index of max element excluding index {i}: {result}")
    print(f"Value at that index: {tensor[result]}")

    # Example usage with floating-point tensor:
    tensor = torch.tensor([5.0594e-04, 9.1007e-01, 1.3037e-05, 4.6771e-05, 6.2311e-07, 9.1759e-06,
                           7.8168e-05, 1.8688e-06, 5.4961e-04, 8.8721e-02])
    i = 1
    start_time = time()
    result = find_max_index_excluding(tensor, i)
    print(f"\nTime taken: {time() - start_time} seconds")
    print(f"Original tensor: {tensor}")
    print(f"Index to exclude: {i}")
    print(f"Index of max element excluding index {i}: {result}")
    print(f"Value at that index: {tensor[result]}")


if __name__ == "__main__":
    main()
