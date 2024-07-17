#!/bin/bash

# Activate the virtual environment
source "$HOME/frugal_rs/.venv/bin/activate"

# Set locale to ensure decimal point handling
export LC_NUMERIC="C"
export PYTHONPATH="$HOME/frugal_rs"

sigma=0.12
temperature=10
sigma_str=$(printf "%.2f" "$sigma")
base_classifier_path="$HOME/models/cifar10/resnet110/noise_$sigma_str/checkpoint.pth.tar"
shift="zack"

# Function to run the command for a single N value
run_command() {
    N=$1
    outfile_path="$HOME/test_results/cifar10_smooth_50_steps/noise_$sigma_str/N_$N/$shift-$temperature.csv"
    log_path="$HOME/test_results/cifar10_smooth_50_steps/noise_$sigma_str/N_$N/log_$shift-$temperature.log"

    # Create necessary directories
    mkdir -p "$(dirname "$outfile_path")"
    mkdir -p "$(dirname "$log_path")"

    command="python certify_comparison.py \
    --base_classifier \"$base_classifier_path\" \
    --temperature \"$temperature\" \
    --n \"$N\" \
    --shift \"$shift\" \
    --outfile \"$outfile_path\" \
    --log \"$log_path\""

    echo "$command"
    eval "$command"
}

# Set the maximum number of parallel jobs
max_jobs=32

# Loop through N values and run commands in parallel
for N in $(seq 100 50 1000)
do
    # Run the command in the background
    run_command $N &

    # Limit the number of parallel jobs
    if (( $(jobs -p | wc -l) >= max_jobs )); then
        wait -n
    fi
done

# Wait for all background jobs to finish
wait

# Deactivate the virtual environment
deactivate