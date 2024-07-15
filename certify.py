from time import time
from typing import Callable, Tuple

from pandas import DataFrame
from scipy.stats import norm
from torch import Tensor, softmax, nn, mean, device, cuda, load
from torch.utils.data import DataLoader

from architectures import get_architecture
from calculate_term import calculate_term
from datasets import get_dataset
from training import smoothed_predict


def certify(model: nn.Module, x: Tensor, n0: int, n: int, noise_sd: float, alpha: float,
            normalize_fn: Callable[[Tensor], Tensor] = softmax) -> Tuple[int, float]:
    first_predictions = smoothed_predict(model, x, n0, noise_sd, normalize_fn)
    first_means = mean(first_predictions, dim=0)
    predicted = first_means.argmax().item()

    second_predictions = smoothed_predict(model, x, n, noise_sd, normalize_fn)
    predicted_class_scores = second_predictions[:, predicted]
    term = calculate_term(predicted_class_scores, alpha)
    predicted_class_score = mean(predicted_class_scores).item()

    p1 = predicted_class_score - term
    certified_radius = noise_sd * norm.ppf(p1) if 0.5 < p1 < 1 else 0.

    return predicted, certified_radius


def main() -> None:
    torch_device = device('cuda' if cuda.is_available() else 'cpu')
    epochs = 5
    noise_sd = 0.12
    n0 = 10
    n = 100
    alpha = 0.001
    print_freq = 1000
    num_workers = 4

    model = get_architecture(torch_device)
    checkpoint = load('saved_models/model_epoch_' + str(epochs) + '.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    train_dataset = get_dataset('train')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=num_workers)
    results = []

    for i, (image, label) in enumerate(train_loader):
        image, label = image.to(torch_device), label.to(torch_device)
        start_time = time()
        prediction, radius = certify(model, image, n0, n, noise_sd, alpha)
        end_time = time()
        results.append({
            'idx': i,
            'label': label,
            'predicted': prediction,
            'correct': int(prediction == label),
            'radius': radius,
            'time': f"{end_time - start_time:.4f}"
        })

        if i % print_freq == 0:
            print(f"Certified Radius: {radius:.4f}, Time: {end_time - start_time:.4f}")

    test_dataset = get_dataset('test')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=num_workers)

    for i, (image, label) in enumerate(test_loader):
        image, label = image.to(torch_device), label.to(torch_device)
        start_time = time()
        prediction, radius = certify(model, image, n0, n, noise_sd, alpha)
        end_time = time()
        results.append({
            'idx': i + len(train_dataset),
            'label': label,
            'predicted': prediction,
            'correct': int(prediction == label),
            'radius': radius,
            'time': f"{end_time - start_time:.4f}"
        })

        if i % print_freq == 0:
            print(f"Certified Radius: {radius:.4f}, Time: {end_time - start_time:.4f}")

    # Create DataFrame from results
    df = DataFrame(results)

    # Save results to CSV
    df.to_csv('logs/certify.csv', index=False)


if __name__ == "__main__":
    main()
