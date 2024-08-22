from logging import Logger
from pathlib import Path

import h5py
import numpy as np
from torch import Tensor, device, from_numpy


class PredictionsPersistence:
    def __init__(self, title: str, output_file: str, max_num_predictions: int, num_examples: int, num_classes: int,
                 logger: Logger) -> None:
        self.title = title
        self.output_file = Path(output_file)
        self.max_num_predictions = max_num_predictions
        self.num_examples = num_examples
        self.num_classes = num_classes
        self.logger = logger

    def first_time(self) -> None:
        with h5py.File(self.output_file, 'x') as f:
            # Create a dataset for storing predictions
            f.create_dataset(f"{self.title}_predictions",
                             shape=(self.num_examples, self.max_num_predictions, self.num_classes),
                             dtype=np.float32,
                             chunks=True,
                             maxshape=(self.num_examples, None, self.num_classes))

            # Create a dataset for storing the count of predictions per example
            f.create_dataset(f"{self.title}_counts",
                             shape=(self.num_examples,),
                             dtype=np.int32,
                             chunks=True)

            # Initialize all counts to 0
            f[f"{self.title}_counts"][:] = 0

        self.logger.info(f"Created initial predictions dataset at {self.output_file}")

    def get_num_predictions(self, example_index: int) -> int:
        with h5py.File(self.output_file, 'r') as f:
            return f[f"{self.title}_counts"][example_index]

    def load_predictions(self, example_index: int, torch_device: device) -> Tensor:
        with h5py.File(self.output_file, 'r') as f:
            count = f[f"{self.title}_counts"][example_index]
            if count == 0:
                raise ValueError(f"No predictions found for example index {example_index}")

            predictions = f[f"{self.title}_predictions"][example_index, :count]

            # Convert to PyTorch tensor
            return from_numpy(predictions).to(torch_device)

    def save_predictions(self, predictions: Tensor, example_index: int) -> None:
        if predictions.dim() != 2 or predictions.shape[1] != self.num_classes:
            raise ValueError(f"Expected predictions shape (n, {self.num_classes}), got {predictions.shape}")

        with h5py.File(self.output_file, 'r+') as f:
            current_count = f[f"{self.title}_counts"][example_index]
            new_count = current_count + len(predictions)

            # If we need more space, resize the dataset
            if new_count > f[f"{self.title}_predictions"].shape[1]:
                new_max = max(new_count, f[f"{self.title}_predictions"].shape[1] + self.max_num_predictions)
                f[f"{self.title}_predictions"].resize((self.num_examples, new_max, self.num_classes))

            # Save the new predictions
            f[f"{self.title}_predictions"][example_index, current_count:new_count] = predictions.numpy()

            # Update the count
            f[f"{self.title}_counts"][example_index] = new_count

        self.logger.info(
            f"Saved {len(predictions)} new predictions for example {example_index} in {self.output_file}"
        )
