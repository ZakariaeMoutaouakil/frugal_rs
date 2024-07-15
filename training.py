import os
from typing import Callable
from pandas import DataFrame
from scipy.stats import norm
from torch import cuda, no_grad, device, save, randn_like, Tensor, nn, argmax, mean
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim import SGD
from torch.utils.data import DataLoader

from architectures import get_architecture
from calculate_term import calculate_term
from datasets import get_dataset
from logging_config import basic_logger


def smoothed_predict(model: nn.Module, x: Tensor, num_samples: int, noise_sd: float,
                     normalize_fn: Callable[[Tensor], Tensor] = softmax) -> Tensor:
    with no_grad():
        # Create noisy inputs by repeating x and adding noise
        noisy_inputs = x.repeat(num_samples, 1, 1, 1) \
                       + randn_like(x.repeat(num_samples, 1, 1, 1), device=x.device) * noise_sd
        # Get model outputs for all noisy inputs
        outputs = model(noisy_inputs)
        # Normalize the outputs using the provided normalization function
        return normalize_fn(outputs)


def main():
    train_batch = 256
    test_batch = 1
    num_workers = 4
    torch_device = device('cuda' if cuda.is_available() else 'cpu')
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 5
    noise_sd = 0.12
    num_noise_samples = 100
    alpha = 0.001
    log_freq = 1000

    logger = basic_logger('logs/training.log')
    results = []

    # Create a directory to save models if it doesn't exist
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = get_dataset('train')
    test_dataset = get_dataset('test')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=test_batch, num_workers=num_workers)

    model = get_architecture(torch_device)

    criterion = CrossEntropyLoss().to(torch_device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(torch_device), labels.to(torch_device)

            # Add Gaussian noise for data augmentation
            noisy_inputs = inputs + randn_like(inputs, device=torch_device) * noise_sd

            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % log_freq == 0:
                logger.debug(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.

        # Save the model after each epoch
        save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
        save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
        }, save_path)
        logger.info(f'Saved model to {save_path}')

        # Evaluate smoothed model after each epoch
        model.eval()
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, true_label = inputs.to(torch_device), labels.to(torch_device).item()
            smoothed_output = smoothed_predict(model, inputs, num_noise_samples, noise_sd)
            means = smoothed_output.mean(dim=0)
            predicted = argmax(means).item()

            correct_class_scores = smoothed_output[:, true_label]
            term = calculate_term(correct_class_scores, alpha)
            correct_class_score = mean(correct_class_scores).item()
            p1 = correct_class_score - term
            certified_radius = noise_sd * norm.ppf(p1) if 0.5 < p1 < 1 else 0.

            results.append({
                'epoch': epoch + 1,
                'idx': i,
                'label': true_label,
                'predicted': predicted,
                'correct': int(predicted == true_label),
                'radius': certified_radius
            })

            if i % log_freq == 0:
                logger.debug("inputs: {}".format(inputs))
                logger.debug("shape: {}".format(inputs.shape))
                logger.debug("smoothed_output: {}".format(smoothed_output))
                logger.debug("means: {}".format(means))
                logger.debug("predicted: {}".format(predicted))
                logger.debug("correct_class_scores: {}".format(correct_class_scores))
                logger.debug("term: {}".format(term))
                logger.debug("correct_class_score: {}".format(correct_class_score))
                logger.debug("p1: {}".format(p1))
                logger.debug("certified_radius: {}".format(certified_radius))

    logger.info('Finished Training and Evaluation')

    # Create DataFrame from results
    df = DataFrame(results)

    # Save results to CSV
    df.to_csv('logs/smoothed_model_results.csv', index=False)

    logger.info('Results saved to smoothed_model_results.csv')


if __name__ == "__main__":
    main()
