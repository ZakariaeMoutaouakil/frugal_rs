#!/bin/bash

# Set locale to ensure decimal point handling
export LC_NUMERIC="C"
export PYTHONPATH="$HOME/PycharmProjects/frugal_rs"

# Activate the virtual environment
source "$PYTHONPATH/.venv/bin/activate"

# Define the array for sigma values
sigmas=(0.12 0.50 1.00)

num_samples_array=(20 50 100)
# Path to the dataset directory
dataset_path="/home/pc/PycharmProjects/test_results/new"
# Output directory
output_dir="/home/pc/PycharmProjects/test_results/transformed_discrete"
mkdir -p "$output_dir"

# Function to process a single combination of sigma and num_samples
process_combination() {
    local sigma=$1
    local num_samples=$2

    # Format sigma to two decimal places
    formatted_sigma=$(printf "%.2f" $sigma)

    # Construct the file path for the dataset
    file_name="cifar10_${formatted_sigma}.h5"
    full_path="${dataset_path}/${file_name}"
    echo "Dataset: $full_path"

    # Construct the output file path
    outdir="${output_dir}/cifar10_${formatted_sigma}_${num_samples}/"
    echo "Output directory: $outdir"

    # Create the output directory if it doesn't exist
    mkdir -p "$outdir"

    # Execute the python command with current loop variables
    python3 -m clopper.certify_dataset --num_samples "$num_samples" --dataset "$full_path" --outdir "$outdir" \
            --max_examples 500
}

# Loop over sigma values and num_samples, running each combination in the background
for sigma in "${sigmas[@]}"; do
    for num_samples in "${num_samples_array[@]}"; do
        process_combination "$sigma" "$num_samples" &
    done
done

# Wait for all background jobs to complete
wait

# Deactivate the virtual environment
deactivate

echo "All tasks completed."
