from argparse import ArgumentParser
from json import dumps
from os.path import basename, splitext
from time import time
from typing import List, Dict

from h5py import File
from pandas import DataFrame
from scipy.stats import norm
from torch import device, mean, from_numpy, set_printoptions
from torch.cuda import is_available as cuda_is_available
from torch.utils.data import DataLoader

from bernstein_optimized.calculate_shift import calculate_shift
from bernstein_optimized.calculate_shift_upper import calculate_shift_upper
from calculate_term import calculate_term
from certify.datasets import get_dataset
from logging_config import basic_logger
from testing.gaussian_quantile_approximation import gaussian_quantile_approximation
from transform.analyze_hdf5_dataset import analyze_hdf5_dataset
from transform.apply_function_to_tensor import apply_function_to_tensor
from transform.find_max_index_excluding import find_max_index_excluding
from transform.softmax_with_temperature import softmax_with_temperature

parser = ArgumentParser(description='Transform dataset')
parser.add_argument("--temperature", type=float, required=True, help="softmax temperature")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--dataset", type=str, help="dataset path", required=True)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--num_samples", type=int, help="number of samples to use", required=True)
parser.add_argument("--order", type=int, help="approximation order", default=15)
parser.add_argument("--outdir", type=str, help="output directory", required=True)
args = parser.parse_args()

args_dict = vars(args)

# Pretty print the dictionary with json.dumps
formatted_args = dumps(args_dict, indent=4)

set_printoptions(threshold=20)


def main() -> None:
    torch_device = device('cuda' if cuda_is_available() else 'cpu')
    dataset_title: str = splitext(basename(args.dataset))[0]
    dataset_type: str = dataset_title.split('_')[0]
    test_dataset = get_dataset(dataset_type)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=args.num_workers)

    dataset_info = analyze_hdf5_dataset(args.dataset)
    last_nonzero_index = dataset_info['last_nonzero_index']
    min_inference_count = dataset_info['min_inference_count']

    if args.num_samples > min_inference_count:
        raise ValueError(f"num_samples ({args.num_samples}) > min_inference_count ({min_inference_count})")

    if last_nonzero_index == -1:
        raise ValueError(f"no non-zero examples found in {args.dataset}")

    dataset = File(args.dataset, 'r')

    logger = basic_logger(args.outdir + '/' + dataset_title + '.log')

    # Log the formatted arguments
    logger.info(formatted_args)
    logger.info(f"torch_device: {torch_device}")
    logger.debug(f"last_nonzero_index: {last_nonzero_index}")
    logger.debug(f"min_inference_count: {min_inference_count}")
    logger.debug(f"dataset_title: {dataset_title}")
    logger.debug(f"dataset_type: {dataset_type}")

    results_bernstein_first: List[Dict[str, str | float]] = []
    results_bernstein_bonferroni_first: List[Dict[str, str | float]] = []
    results_sequence_first: List[Dict[str, str | float]] = []
    results_sequence_bonferroni_first: List[Dict[str, str | float]] = []

    results_bernstein_bonferroni_second: List[Dict[str, str | float]] = []
    results_sequence_second: List[Dict[str, str | float]] = []
    results_sequence_bonferroni_second: List[Dict[str, str | float]] = []

    quantile_function = gaussian_quantile_approximation(args.order)
    maximum = quantile_function(1) - quantile_function(0)

    for i, (_, label) in enumerate(test_loader):
        if i >= last_nonzero_index:
            break

        label = label.item()
        logits = from_numpy(dataset[f"{dataset_title}_predictions"][i, :args.num_samples]).to(torch_device)
        logger.debug(f"Example {i} - label: {label} - logits: {logits}")
        predictions = softmax_with_temperature(logits=logits, temperature=args.temperature)
        logger.debug(f"predictions: {predictions}")
        means = mean(predictions, dim=0)
        logger.debug(f"means: {means}")

        predicted = means.argmax().item()
        logger.debug(f"predicted: {predicted}")
        second_class = find_max_index_excluding(tensor=means, i=predicted)
        logger.debug(f"second_class: {second_class}")
        correct = int(predicted == label)
        logger.debug(f"correct: {correct}")

        predicted_tensor = predictions[:, predicted]
        second_tensor = predictions[:, second_class]
        difference_tensor = predicted_tensor - second_tensor
        normalized_difference_tensor = (difference_tensor + 1) / 2
        logger.debug(f"predicted_tensor: {predicted_tensor}")
        logger.debug(f"second_tensor: {second_tensor}")
        logger.debug(f"difference_tensor: {difference_tensor}")
        logger.debug(f"normalized_difference_tensor: {normalized_difference_tensor}")

        ## First Radius

        # Bernstein
        start_time = time()
        normalized_bernstein_term = calculate_term(vector=normalized_difference_tensor, alpha=args.alpha)
        bernstein_term = 2 * normalized_bernstein_term - 1
        bernstein_lower_bound = mean(difference_tensor).item() - bernstein_term
        end_time = time()
        results_bernstein_first.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., bernstein_lower_bound),
            'time': f"{end_time - start_time:.4f}"
        })
        logger.debug(results_bernstein_first[-1])

        # Bernstein + Bonferroni
        start_time = time()
        predicted_term = calculate_term(vector=predicted_tensor, alpha=args.alpha / 2)
        second_term = calculate_term(vector=second_tensor, alpha=args.alpha / 2)
        predicted_lower_bound = mean(predicted_tensor).item() - predicted_term
        second_upper_bound = mean(second_tensor).item() + second_term
        bernstein_bonferroni_lower_bound = predicted_lower_bound - second_upper_bound
        end_time = time()
        results_bernstein_bonferroni_first.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., bernstein_bonferroni_lower_bound),
            'time': f"{end_time - start_time:.4f}"
        })
        logger.debug(results_bernstein_bonferroni_first[-1])

        # Sequence
        start_time = time()
        normalized_lower_bound = calculate_shift(x=normalized_difference_tensor, alpha_0=args.alpha)
        sequence_lower_bound = 2 * normalized_lower_bound - 1
        end_time = time()
        results_sequence_first.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., sequence_lower_bound),
            'time': f"{end_time - start_time:.4f}"
        })
        logger.debug(results_sequence_first[-1])

        # Sequence + Bonferroni
        start_time = time()
        predicted_sequence_lb = calculate_shift(x=predicted_tensor, alpha_0=args.alpha / 2)
        second_sequence_up = calculate_shift_upper(x=second_tensor, alpha_0=args.alpha / 2)
        sequence_bonferroni_lb = predicted_sequence_lb - second_sequence_up
        end_time = time()
        results_sequence_bonferroni_first.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., sequence_bonferroni_lb),
            'time': f"{end_time - start_time:.4f}"
        })
        logger.debug(results_sequence_bonferroni_first[-1])

        ## Second Radius

        # Bernstein + Bonferroni
        start_time = time()
        predicted_term_second = calculate_term(vector=predicted_tensor, alpha=args.alpha / 2)
        second_class_term_second = calculate_term(vector=second_tensor, alpha=args.alpha / 2)
        predicted_lower_bound_second = mean(predicted_tensor).item() - predicted_term_second
        second_class_upper_bound_second = mean(second_tensor).item() + second_class_term_second
        bernstein_bonferroni_lower_bound_second = norm.ppf(predicted_lower_bound_second) - norm.ppf(
            second_class_upper_bound_second)
        end_time = time()
        results_bernstein_bonferroni_second.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., bernstein_bonferroni_lower_bound_second),
            'time': f"{end_time - start_time:.4f}"
        })
        logger.debug(results_bernstein_bonferroni_second[-1])

        # Sequence
        start_time = time()
        predicted_class_lb_term = calculate_shift(x=predicted_tensor, alpha_0=args.alpha / 2)
        predicted_class_lb = mean(predicted_tensor).item() - predicted_class_lb_term

        if predicted_class_lb >= 1 / 2:
            predicted_quantiles = apply_function_to_tensor(x=predicted_tensor, func=quantile_function)
            second_class_quantiles = apply_function_to_tensor(x=second_tensor, func=quantile_function)
            difference_quantiles = predicted_quantiles - second_class_quantiles
            difference_quantiles_normalized = difference_quantiles / maximum
            difference_quantiles_normalized_lb = calculate_shift(x=difference_quantiles_normalized,
                                                                 alpha_0=args.alpha / 2)
            sequence_lower_bound_second = difference_quantiles_normalized_lb * maximum
            end_time = time()
            results_sequence_second.append({
                'idx': i,
                'label': label,
                'predicted': predicted,
                'correct': correct,
                'radius': max(0., sequence_lower_bound_second),
                'time': f"{end_time - start_time:.4f}"
            })
            logger.debug(results_sequence_second[-1])
        else:
            predicted_sequence_lb_second = calculate_shift(x=predicted_tensor, alpha_0=args.alpha / 2)
            second_sequence_up_second = calculate_shift_upper(x=second_tensor, alpha_0=args.alpha / 2)
            sequence_bonferroni_lb_second = norm.ppf(predicted_sequence_lb_second) - norm.ppf(second_sequence_up_second)
            end_time = time()
            results_sequence_bonferroni_second.append({
                'idx': i,
                'label': label,
                'predicted': predicted,
                'correct': correct,
                'radius': max(0., sequence_bonferroni_lb_second),
                'time': f"{end_time - start_time:.4f}"
            })

        # Sequence + Bonferroni
        start_time = time()
        predicted_sequence_lb_second = calculate_shift(x=predicted_tensor, alpha_0=args.alpha / 2)
        second_sequence_up_second = calculate_shift_upper(x=second_tensor, alpha_0=args.alpha / 2)
        sequence_bonferroni_lb_second = norm.ppf(predicted_sequence_lb_second) - norm.ppf(second_sequence_up_second)
        end_time = time()
        results_sequence_bonferroni_second.append({
            'idx': i,
            'label': label,
            'predicted': predicted,
            'correct': correct,
            'radius': max(0., sequence_bonferroni_lb_second),
            'time': f"{end_time - start_time:.4f}"
        })

    df_bernstein_first = DataFrame(results_bernstein_first)
    df_bernstein_bonferroni_first = DataFrame(results_bernstein_bonferroni_first)
    df_sequence_first = DataFrame(results_sequence_first)
    df_sequence_bonferroni_first = DataFrame(results_sequence_bonferroni_first)

    df_bernstein_bonferroni_second = DataFrame(results_bernstein_bonferroni_second)
    df_sequence_second = DataFrame(results_sequence_second)
    df_sequence_bonferroni_second = DataFrame(results_sequence_bonferroni_second)

    df_bernstein_first.to_csv(args.outdir + '/' + dataset_title + '_bernstein_first.csv', index=False)
    df_bernstein_bonferroni_first.to_csv(args.outdir + '/' + dataset_title + '_bernstein_bonferroni_first.csv',
                                         index=False)
    df_sequence_first.to_csv(args.outdir + '/' + dataset_title + '_sequence_first.csv', index=False)
    df_sequence_bonferroni_first.to_csv(args.outdir + '/' + dataset_title + '_sequence_bonferroni_first.csv',
                                        index=False)

    df_bernstein_bonferroni_second.to_csv(
        args.outdir + '/' + dataset_title + '_bernstein_bonferroni_second.csv',
        index=False)
    df_sequence_second.to_csv(args.outdir + '/' + dataset_title + '_sequence_second.csv', index=False)
    df_sequence_bonferroni_second.to_csv(args.outdir + '/' + dataset_title + '_sequence_bonferroni_second.csv',
                                         index=False)

    logger.info(f"Saved results to {args.outdir} directory")


if __name__ == "__main__":
    main()
