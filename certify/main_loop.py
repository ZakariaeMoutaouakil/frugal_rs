from argparse import ArgumentParser
from json import dumps

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
    persistence = PredictionsPersistence(f"{args.dataset}_{args.sigma}", args.outfile, args.num_samples,
                                         len(test_dataset), num_classes, logger)
    persistence.first_time()

    early_stop = False

    try:
        for i, (image, _) in enumerate(test_loader):
            if early_stop:
                break

            try:
                image = image.to(torch_device)
                logits = smoothed_logits(model, image, args.num_samples, args.sigma)
                logger.debug(f"Predictions: {logits}")
                persistence.save_predictions(logits, i)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected. Finishing current iteration...")
                early_stop = True

    finally:
        logger.info("Finished processing. Cleaning up...")

    logger.info("Script completed.")


if __name__ == "__main__":
    main()
