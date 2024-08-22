from logging import Logger
from pathlib import Path

import h5py
import numpy as np
from torch import Tensor, device, from_numpy


class PredictionsPersistence:
    def __init__(self, title: str, output_file: str, max_num_predictions: int, num_examples: int, num_classes: int,
                 logger: Logger, unused_flag: float = np.nan) -> None:
        self.title = title
        self.output_file = Path(output_file)
        self.max_num_predictions = max_num_predictions
        self.num_examples = num_examples
        self.num_classes = num_classes
        self.logger = logger
        self.unused_flag = unused_flag  # We'll use NaN to indicate unused prediction slots

    def first_time(self) -> None:
        with h5py.File(self.output_file, 'x') as f:
            # Create a dataset for storing predictions
            f.create_dataset(self.title,
                             shape=(self.num_examples, self.max_num_predictions, self.num_classes),
                             dtype=np.float32,
                             chunks=True,
                             maxshape=(self.num_examples, None, self.num_classes))

            # Initialize all prediction slots to our unused flag
            f[self.title][:] = self.unused_flag

        self.logger.info(f"Created initial predictions dataset at {self.output_file}")

    def get_num_predictions(self, example_index: int) -> int:
        with h5py.File(self.output_file, 'r') as f:
            predictions = f[self.title][example_index]
            # Count the number of prediction slots that are not our unused flag
            return np.sum(~np.isnan(predictions[:, 0]))  # We only need to check the first class

    def load_predictions(self, example_index: int, torch_device: device) -> Tensor:
        with h5py.File(self.output_file, 'r') as f:
            predictions = f[self.title][example_index]

            # Find valid predictions (not equal to unused_flag)
            valid_mask = ~np.isnan(predictions[:, 0])  # Check only the first class
            valid_predictions = predictions[valid_mask]

            if len(valid_predictions) == 0:
                raise ValueError(f"No predictions found for example index {example_index}")

            # Convert to PyTorch tensor
            return from_numpy(valid_predictions).to(torch_device)

    def save_predictions(self, predictions: Tensor, example_index: int) -> None:
        if predictions.dim() != 2 or predictions.shape[1] != self.num_classes:
            raise ValueError(f"Expected predictions shape (n, {self.num_classes}), got {predictions.shape}")

        with h5py.File(self.output_file, 'r+') as f:
            existing_predictions = f[self.title][example_index]

            # Find the first unused slot
            first_unused = np.argmax(np.isnan(existing_predictions[:, 0]))

            # If no unused slots, we need to resize the dataset
            if first_unused == 0 and not np.isnan(existing_predictions[0, 0]):
                current_max = existing_predictions.shape[0]
                new_max = current_max + self.max_num_predictions
                f[self.title].resize((self.num_examples, new_max, self.num_classes))
                existing_predictions = f[self.title][example_index]
                existing_predictions[current_max:] = self.unused_flag
                first_unused = current_max

            # Calculate how many new predictions we can fit
            space_available = len(existing_predictions) - first_unused
            predictions_to_save = min(len(predictions), space_available)

            # Save the new predictions
            existing_predictions[first_unused:first_unused + predictions_to_save] = predictions[
                                                                                    :predictions_to_save].numpy()
            f[self.title][example_index] = existing_predictions

        self.logger.info(
            f"Saved {predictions_to_save} new predictions for example {example_index} in {self.output_file}"
        )

        # Log a warning if we couldn't save all predictions
        if predictions_to_save < len(predictions):
            self.logger.warning(
                f"{len(predictions) - predictions_to_save} predictions could not be saved due to lack of space"
            )
