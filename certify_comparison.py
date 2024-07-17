from argparse import ArgumentParser
from json import dumps
from time import time
from typing import Callable, Tuple

from pandas import DataFrame
from scipy.stats import norm
from torch import Tensor, nn, mean, device, cuda, load
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from architectures import get_architecture
from bernstein.calculate_shift import calculate_shift
from calculate_term import calculate_term
from datasets import get_dataset
from logging_config import basic_logger
from training import smoothed_predict

parser = ArgumentParser(description='Certify many examples')
parser.add_argument("--base_classifier", type=str, help="path to saved pytorch model of base classifier",
                    required=True)
parser.add_argument("--sigma", type=float, default=0.12, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file", required=True)
parser.add_argument("--temperature", type=float, default=1.0, help="softmax temperature")
parser.add_argument("--max", type=int, default=1000, help="stop after this many examples")
parser.add_argument("--n0", type=int, default=100)
parser.add_argument("--n", type=int, help="number of samples to use", required=True)
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--log", type=str, help="Location of log file", required=True)
parser.add_argument("--shift", type=str, choices=["blaise", "zack"], required=True)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--print_freq", type=int, default=100)
args = parser.parse_args()

logger = basic_logger(args.log)
args_dict = vars(args)

# Pretty print the dictionary with json.dumps
formatted_args = dumps(args_dict, indent=4)

# Log the formatted arguments
logger.info(formatted_args)


def certify(model: nn.Module, x: Tensor, n0: int, n: int, noise_sd: float, alpha: float, temperature: float,
            shift: str, debug: bool) \
        -> Tuple[int, float]:
    normalize_fn: Callable[[Tensor], Tensor] = lambda y: softmax(y / temperature, dim=1)
    first_predictions = smoothed_predict(model, x, n0, noise_sd, normalize_fn)
    first_means = mean(first_predictions, dim=0)
    predicted = first_means.argmax().item()

    second_predictions = smoothed_predict(model, x, n, noise_sd, normalize_fn)
    predicted_class_scores = second_predictions[:, predicted]

    p1 = 0.
    if shift == 'blaise':
        term = calculate_term(predicted_class_scores, alpha)
        predicted_class_score = mean(predicted_class_scores).item()
        p1 = predicted_class_score - term
    elif shift == 'zack':
        p1 = calculate_shift(predicted_class_scores, alpha)

    certified_radius = noise_sd * norm.ppf(p1) if 0.5 < p1 < 1 else 0.
    if debug:
        logger.debug(f"normalize_fn: {normalize_fn}")
        logger.debug(f"first_predictions: {first_predictions}")
        logger.debug(f"first_means: {first_means}")
        logger.debug(f"predicted: {predicted}")
        logger.debug(f"second_predictions: {second_predictions}")
        logger.debug(f"predicted_class_scores: {predicted_class_scores}")
        logger.debug(f"p1: {p1}")
    return predicted, certified_radius


def main() -> None:
    torch_device = device('cuda' if cuda.is_available() else 'cpu')
    checkpoint = load(args.base_classifier, map_location=torch_device)
    model = get_architecture(torch_device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(torch_device)
    model.eval()

    results = []

    test_dataset = get_dataset('test')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=args.num_workers)

    for i, (image, label) in enumerate(test_loader):
        if i >= args.max:
            break

        image, label = image.to(torch_device), label.to(torch_device).item()
        start_time = time()
        prediction, radius = certify(model=model, x=image, n0=args.n0, n=args.n, noise_sd=args.sigma, alpha=args.alpha,
                                     temperature=args.temperature, shift=args.shift, debug=(i % args.print_freq == 0))
        end_time = time()
        results.append({
            'idx': i,
            'label': label,
            'predicted': prediction,
            'correct': int(prediction == label),
            'radius': radius,
            'time': f"{end_time - start_time:.4f}"
        })

        logger.debug(f"idx: {i}")
        logger.debug(f"label: {label}")
        logger.debug(f"predicted: {prediction}")
        logger.debug(f"correct: {int(prediction == label)}")
        logger.debug(f"radius: {radius}")
        logger.debug(f"time: {end_time - start_time:.4f}")

    # Create DataFrame from results
    df = DataFrame(results)

    # Save results to CSV
    df.to_csv(args.outfile, index=False)


if __name__ == "__main__":
    main()
