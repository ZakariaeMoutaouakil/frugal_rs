#!/bin/bash

# Activate the virtual environment
source "$HOME/frugal_rs/.venv/bin/activate"

# Set locale to ensure decimal point handling
export LC_NUMERIC="C"
export PYTHONPATH="$HOME/frugal_rs"

sigma=0.12
temperature=1.0
sigma_str=$(printf "%.2f" "$sigma")
base_classifier_path="$HOME/models/cifar10/resnet110/noise_$sigma_str/checkpoint.pth.tar"
shift="blaise"

for N in $(seq 100 10 1000)
do
    outfile_path="$HOME/test_results/cifar10_smoothed/noise_$sigma_str/N_$N/$shift-$temperature.csv"
    log_path="$HOME/test_results/cifar10_smoothed/noise_$sigma_str/N_$N/log_$shift-$temperature.log"

    # Create necessary directories
    mkdir -p "$(dirname "$outfile_path")"
    mkdir -p "$(dirname "$log_path")"

    command="python certify_comparison.py\
    --base_classifier \"$base_classifier_path\" \
    --temperature \"$temperature\" \
    --n \"$N\" \
    --shift \"$shift\" \
    --outfile \"$outfile_path\" \
    --log \"$log_path\""

    echo "$command"
    eval "$command"
done

# Deactivate the virtual environment
deactivate
