from argparse import ArgumentParser
from csv import DictReader
from json import dumps
from os import listdir
from os.path import splitext, basename
from time import time
from typing import Dict, List, Optional, Tuple

from h5py import File
from pandas import DataFrame
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from torch import device, from_numpy, set_printoptions
from torch.cuda import is_available as cuda_is_available
from torch.utils.data import DataLoader

from certify.datasets import get_dataset
from clopper.calculate_coefficients import calculate_coefficients
from clopper.count_column_maxima import count_column_maxima
from clopper.dichotomy import dichotomy
from logging_config import basic_logger
from testing.gaussian_quantile_approximation import gaussian_quantile_approximation
from transform.analyze_hdf5_dataset import analyze_hdf5_dataset
from transform.find_max_index_excluding import find_max_index_excluding


def get_last_idx(file_path: str) -> Optional[int]:
    last_idx: Optional[int] = None

    with open(file_path, 'r') as csvfile:
        csv_reader = DictReader(csvfile)
        for row in csv_reader:
            last_idx = int(row['idx'])

    return last_idx


def is_folder_empty(folder_path: str) -> bool:
    """Check if a folder is empty by verifying the number of items in it."""
    # Get the list of files and directories in the folder
    if not listdir(folder_path):
        return True
    else:
        return False


parser = ArgumentParser(description='Certify dataset')
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--dataset", type=str, help="dataset path", required=True)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--num_samples", type=int, help="number of samples to use", required=True)
parser.add_argument("--order", type=int, help="approximation order", default=15)
parser.add_argument("--outdir", type=str, help="output directory", required=True)
parser.add_argument("--max_examples", type=int, help="maximum number of examples to certify", required=True)
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

    last_idx = -1
    try:
        last_idx = get_last_idx(args.outdir + '/' + dataset_title + '_cp_first.csv')
    except FileNotFoundError:
        pass

    dataset = File(args.dataset, 'r')

    logger = basic_logger(args.outdir + '/' + dataset_title + '.log')

    # Log the formatted arguments
    logger.info(formatted_args)
    logger.info(f"torch_device: {torch_device}")
    logger.debug(f"last_nonzero_index: {last_nonzero_index}")
    logger.debug(f"min_inference_count: {min_inference_count}")
    logger.debug(f"dataset_title: {dataset_title}")
    logger.debug(f"dataset_type: {dataset_type}")
    logger.debug(f"last_idx: {last_idx}")

    results_cp_first: List[Dict[str, float | str]] = []
    results_dichotomy_first: List[Dict[str, float | str]] = []
    results_cp_second: List[Dict[str, float | str]] = []
    results_dichotomy_second: List[Dict[str, float | str]] = []

    quantile_function = gaussian_quantile_approximation(args.order)
    coefficients = calculate_coefficients(args.num_samples)
    global_time = time()

    cached_results_first: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
    cached_results_second: Dict[Tuple[int, int, int], Tuple[float, float]] = {}

    early_stop = False
    try:
        for i, (_, label) in enumerate(test_loader):
            if i == last_nonzero_index:
                break
            if i <= last_idx:
                continue
            if i >= args.max_examples:
                break
            if early_stop:
                break

            try:
                label = label.item()
                logits = from_numpy(dataset[f"{dataset_title}_predictions"][i, :args.num_samples]).to(torch_device)
                logger.debug(f"Example {i} - label: {label} - logits: {logits}")
                counts = count_column_maxima(logits)
                logger.debug(f"counts: {counts}")

                predicted = counts.argmax().item()
                logger.debug(f"predicted: {predicted}")
                second_class = find_max_index_excluding(tensor=counts, i=predicted)
                logger.debug(f"second_class: {second_class}")
                correct = int(predicted == label)
                logger.debug(f"correct: {correct}")

                predicted_count = counts[predicted].item()
                second_class_count = counts[second_class].item()
                reduced_counts = (
                    predicted_count, second_class_count, args.num_samples - predicted_count - second_class_count)
                logger.debug(f"reduced_counts: {reduced_counts}")

                if correct == 0:
                    incorrect = {
                        'idx': i,
                        'label': label,
                        'predicted': predicted,
                        'second_class': second_class,
                        'correct': correct,
                        'radius': 0.,
                        'time': 0.
                    }

                    results_cp_first.append(incorrect)
                    results_dichotomy_first.append(incorrect)
                    results_cp_second.append(incorrect)
                    results_dichotomy_second.append(incorrect)
                    logger.info(f"Incorrect: {incorrect}")

                else:
                    ## First Radius
                    logger.info("First Radius")

                    # Clopper-Pearson + Bonferroni
                    start_time = time()
                    predicted_cp_lb = proportion_confint(predicted_count, args.num_samples, alpha=args.alpha / 2,
                                                         method="beta")[0]
                    second_cp_ub = proportion_confint(second_class_count, args.num_samples, alpha=args.alpha / 2,
                                                      method="beta")[1]
                    radius_cp_first = predicted_cp_lb - second_cp_ub
                    results_cp_first.append({
                        'idx': i,
                        'label': label,
                        'predicted': predicted,
                        'second_class': second_class,
                        'correct': correct,
                        'radius': max(0., radius_cp_first),
                        'time': f"{time() - start_time:.4f}"
                    })
                    logger.info("Clopper-Pearson + Bonferroni")
                    logger.debug(results_cp_first[-1])

                    # Dichotomy
                    if reduced_counts in cached_results_first:
                        logger.debug(f"cached_results_first[{reduced_counts}]: {cached_results_first[reduced_counts]}")
                        results_dichotomy_first.append({
                            'idx': i,
                            'label': label,
                            'predicted': predicted,
                            'second_class': second_class,
                            'correct': correct,
                            'radius': max(0., cached_results_first[reduced_counts][0]),
                            'time': f"{cached_results_first[reduced_counts][1]:.4f}"
                        })
                    else:
                        start_time = time()
                        predicted_class_lb = proportion_confint(predicted_count, args.num_samples, alpha=args.alpha / 2,
                                                                method="beta")[0]
                        second_class_ub = proportion_confint(second_class_count, args.num_samples, alpha=args.alpha / 2,
                                                             method="beta")[1]
                        lower_bound = predicted_class_lb - second_class_ub
                        observation_first = (predicted_count - second_class_count) / args.num_samples
                        if predicted_class_lb >= 0.5:
                            radius_dichotomy_first = dichotomy(x=reduced_counts, alpha=args.alpha / 2, left=lower_bound,
                                                               right=observation_first, coefficients=coefficients,
                                                               h=lambda x: x)
                            cached_results_first[reduced_counts] = (
                            max(0., radius_dichotomy_first), time() - start_time)
                            results_dichotomy_first.append({
                                'idx': i,
                                'label': label,
                                'predicted': predicted,
                                'second_class': second_class,
                                'correct': correct,
                                'radius': max(0., radius_dichotomy_first),
                                'time': f"{time() - start_time:.4f}"
                            })
                            logger.info("Dichotomy >= 0.5")
                            logger.debug(results_dichotomy_first[-1])
                        else:
                            cached_results_first[reduced_counts] = (max(0., lower_bound), time() - start_time)
                            results_dichotomy_first.append({
                                'idx': i,
                                'label': label,
                                'predicted': predicted,
                                'second_class': second_class,
                                'correct': correct,
                                'radius': max(0., lower_bound),
                                'time': f"{time() - start_time:.4f}"
                            })
                            logger.info("Dichotomy < 0.5")
                            logger.debug(results_dichotomy_first[-1])

                    ## Second Radius
                    logger.info("Second Radius")

                    # Clopper-Pearson
                    start_time = time()
                    predicted_class_cp_lb = proportion_confint(predicted_count, args.num_samples, alpha=args.alpha / 2,
                                                               method="beta")[0]
                    second_class_cp_ub = proportion_confint(second_class_count, args.num_samples, alpha=args.alpha / 2,
                                                            method="beta")[1]
                    radius_cp_second = norm.ppf(predicted_class_cp_lb) - norm.ppf(second_class_cp_ub)
                    results_cp_second.append({
                        'idx': i,
                        'label': label,
                        'predicted': predicted,
                        'second_class': second_class,
                        'correct': correct,
                        'radius': max(0., radius_cp_second),
                        'time': f"{time() - start_time:.4f}"
                    })
                    logger.info("Clopper-Pearson")
                    logger.debug(results_cp_second[-1])

                    # Dichotomy
                    if reduced_counts in cached_results_second:
                        logger.debug(
                            f"cached_results_second[{reduced_counts}]: {cached_results_second[reduced_counts]}")
                        results_dichotomy_second.append({
                            'idx': i,
                            'label': label,
                            'predicted': predicted,
                            'second_class': second_class,
                            'correct': correct,
                            'radius': max(0., cached_results_second[reduced_counts][0]),
                            'time': f"{cached_results_second[reduced_counts][1]:.4f}"
                        })
                    else:
                        start_time = time()
                        predicted_lb = proportion_confint(predicted_count, args.num_samples, alpha=args.alpha / 2,
                                                          method="beta")[0]
                        second_ub = proportion_confint(second_class_count, args.num_samples, alpha=args.alpha / 2,
                                                       method="beta")[1]
                        lb = quantile_function(predicted_lb) - quantile_function(second_ub)
                        observation_second = (quantile_function(predicted_count / args.num_samples)
                                              - quantile_function(second_class_count / args.num_samples))
                        if predicted_lb >= 0.5:
                            radius_dichotomy_second = dichotomy(x=reduced_counts, alpha=args.alpha / 2, left=lb,
                                                                right=observation_second, coefficients=coefficients,
                                                                h=quantile_function)
                            cached_results_second[reduced_counts] = (
                            max(0., radius_dichotomy_second), time() - start_time)
                            results_dichotomy_second.append({
                                'idx': i,
                                'label': label,
                                'predicted': predicted,
                                'second_class': second_class,
                                'correct': correct,
                                'radius': max(0., radius_dichotomy_second),
                                'time': f"{time() - start_time:.4f}"
                            })
                            logger.info("Dichotomy >= 0.5")
                            logger.debug(results_dichotomy_second[-1])
                        else:
                            cached_results_second[reduced_counts] = (max(0., lb), time() - start_time)
                            results_dichotomy_second.append({
                                'idx': i,
                                'label': label,
                                'predicted': predicted,
                                'second_class': second_class,
                                'correct': correct,
                                'radius': max(0., lb),
                                'time': f"{time() - start_time:.4f}"
                            })
                            logger.info("Dichotomy < 0.5")
                            logger.debug(results_dichotomy_second[-1])

            except KeyboardInterrupt:
                logger.info("Interrupted")
                early_stop = True

    except Exception as e:
        logger.error(f'Error: {e}')
    finally:
        df_cp_first = DataFrame(results_cp_first)
        df_dichotomy_first = DataFrame(results_dichotomy_first)
        df_cp_second = DataFrame(results_cp_second)
        df_dichotomy_second = DataFrame(results_dichotomy_second)

        df_cp_first.to_csv(args.outdir + '/' + dataset_title + '_cp_first.csv', index=False, mode='a',
                           header=last_idx == -1)
        df_dichotomy_first.to_csv(args.outdir + '/' + dataset_title + '_dichotomy_first.csv', index=False, mode='a',
                                  header=last_idx == -1)
        df_cp_second.to_csv(args.outdir + '/' + dataset_title + '_cp_second.csv', index=False, mode='a',
                            header=last_idx == -1)
        df_dichotomy_second.to_csv(args.outdir + '/' + dataset_title + '_dichotomy_second.csv', index=False, mode='a',
                                   header=last_idx == -1)

        logger.info("Saved results to " + args.outdir)
        logger.info("Done in {:.4f}s".format(time() - global_time))


if __name__ == '__main__':
    main()
