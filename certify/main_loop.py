from argparse import ArgumentParser
from json import dumps
from time import time

from h5py import File
from numpy import int32, where, zeros
from torch import nn, Tensor, no_grad, randn_like, device, cuda, load
from torch.utils.data import DataLoader

from certify.PredictionsPersistence import PredictionsPersistence
from certify.architectures import get_architecture
from certify.datasets import DATASETS, get_num_classes, get_dataset
from logging_config import basic_logger

parser = ArgumentParser(description='Certify the base classifier over the test dataset')
parser.add_argument("--base_classifier", type=str, help="path to saved pytorch model of base classifier",
                    required=True)
parser.add_argument("--sigma", type=float, default=0.12, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file", required=True)
parser.add_argument("--num_samples", type=int, help="number of samples to use", required=True)
parser.add_argument("--log_file", type=str, help="Location of log file", required=True)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--dataset", choices=DATASETS, help="dataset", required=True)
parser.add_argument("--max_inferences", type=int, default=10000, help="max number of inferences")
args = parser.parse_args()

logger = basic_logger(args.log_file)
args_dict = vars(args)

# Pretty print the dictionary with json.dumps
formatted_args = dumps(args_dict, indent=4)

# Log the formatted arguments
logger.info(formatted_args)


def smoothed_logits(model: nn.Module, x: Tensor, num_samples: int, noise_sd: float) -> Tensor:
    with no_grad():
        # Create noisy inputs by repeating x and adding noise
        y = x.repeat(num_samples, 1, 1, 1)
        noisy_inputs = y + randn_like(y, device=x.device) * noise_sd
        # Get model outputs for all noisy inputs
        logits = model(noisy_inputs)
        return logits


def main():
    torch_device = device('cuda' if cuda.is_available() else 'cpu')
    logger.info(f"torch_device: {torch_device}")

    checkpoint = load(args.base_classifier, map_location=torch_device)
    model = get_architecture(args.dataset, torch_device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(torch_device)
    model.eval()

    test_dataset = get_dataset(args.dataset)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=args.num_workers)

    num_classes = get_num_classes(args.dataset)
    persistence = PredictionsPersistence(title=f"{args.dataset}_{args.sigma:.2f}", output_file=args.outfile,
                                         max_num_predictions=args.max_inferences, num_examples=len(test_dataset),
                                         num_classes=num_classes, logger=logger)

    try:
        persistence.first_time()
        logger.info("Starting from the beginning of the dataset.")
        counts = zeros(len(test_dataset), dtype=int32)
    except FileExistsError:
        logger.info("Existing file found. Determining examples that need more inferences...")
        with File(args.outfile, 'r') as f:
            counts = f[f"{args.dataset}_{args.sigma:.2f}_counts"][:]

    # Find examples that need more inferences
    unfulfilled_indices = where(counts < args.num_samples)[0]
    logger.debug(f"Found {len(unfulfilled_indices)} examples that need more inferences.")

    early_stop = False

    try:
        for i, (image, _) in enumerate(test_loader):
            if i not in unfulfilled_indices:
                continue

            if early_stop:
                break

            try:
                image = image.to(torch_device)
                # Calculate how many more inferences we need
                num_inferences_needed = args.num_samples - counts[i]
                start_time = time()
                logits = smoothed_logits(model, image, num_inferences_needed, args.sigma)
                logger.debug(f"Predictions: {logits}")
                logger.debug(f"Processing example {i}. Adding {num_inferences_needed} inferences. "
                             f"Predictions shape: {logits.shape}")
                logger.debug(f"Example {i} took {time() - start_time:.2f} seconds.")
                persistence.save_predictions(logits, i)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected. Finishing current iteration...")
                early_stop = True

            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

    finally:
        logger.info("Finished processing. Cleaning up...")

    logger.info("Script completed.")


if __name__ == "__main__":
    main()
