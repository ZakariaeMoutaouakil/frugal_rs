#!/bin/bash

# Set locale to ensure decimal point handling
export LC_NUMERIC="C"
export PYTHONPATH="$HOME/PycharmProjects/frugal_rs"

# Activate the virtual environment
source "$PYTHONPATH/.venv/bin/activate"

# Define the array for sigma values
sigmas=(0.12)
# Define the array for temperature values
temperatures=(1)
# Path to the dataset directory
dataset_path="/home/pc/PycharmProjects/test_results/new"
# Output directory
output_dir="/home/pc/PycharmProjects/test_results/transformed_new"
mkdir -p "$output_dir"

# Loop over sigma values
for sigma in "${sigmas[@]}"; do
    # Format sigma to two decimal places
    formatted_sigma=$(printf "%.2f" $sigma)

    # Loop over temperature values
    for temperature in "${temperatures[@]}"; do
        # Loop over num_samples from 100 to 1000 with a step of 100
        for num_samples in {100..200..100}; do
            # Construct the file path for the dataset
            file_name="cifar10_${formatted_sigma}.h5"
            full_path="${dataset_path}/${file_name}"
            echo "Dataset: $full_path"

            # Construct the output file path
            outdir="${output_dir}/cifar10_${formatted_sigma}_${num_samples}_${temperature}/"
            echo "Output directory: $outdir"

            # Create the output directory if it doesn't exist
            mkdir -p "$outdir"

            # Execute the python command with current loop variables
            python3 -m transform.transform_dataset --temperature "$temperature" --num_samples "$num_samples" --dataset "$full_path" --outdir "$outdir"
        done
    done
done

# Deactivate the virtual environment
deactivate
