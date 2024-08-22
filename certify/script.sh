#!/bin/bash

# Activate the virtual environment
source "$HOME/frugal_rs/.venv/bin/activate"

# Set locale to ensure decimal point handling
export LC_NUMERIC="C"
export PYTHONPATH="$HOME/frugal_rs"
export IMAGENET_DIR="$HOME/imagenet"

usage() {
    echo "Usage: $0 --num_samples <number> --sigma <number> (-i | -c)" >&2
    exit 1
}

# Parse long options
options=$(getopt -o 'ic' -l 'num_samples:,sigma:' -- "$@")
[ $? -eq 0 ] || usage
eval set -- "$options"

# Initialize variables
num_samples=""
sigma=""
option=""

# Parse all options
while true; do
    case "$1" in
        --num_samples) num_samples=$2; shift 2;;
        --sigma) sigma=$2; shift 2;;
        -i)
            if [ "$option" = "c" ]; then
                echo "Error: Options -i and -c are mutually exclusive." >&2
                exit 1
            fi
            option="i"; shift;;
        -c)
            if [ "$option" = "i" ]; then
                echo "Error: Options -i and -c are mutually exclusive." >&2
                exit 1
            fi
            option="c"; shift;;
        --) shift; break;;
        *) usage;;
    esac
done

# Check if required options are present and valid
if ! [[ "$num_samples" =~ ^[0-9]+$ ]] || ! [[ "$sigma" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: num_samples and sigma must be valid numbers" >&2
    usage
fi

if [ -z "$option" ]; then
    echo "Error: One of -i or -c must be specified." >&2
    usage
fi

# Convert sigma to a string with two decimal places
sigma_str=$(printf "%.2f" "$sigma")

# Set the base_classifier_path based on the option
if [ "$option" = "i" ]; then
  base_classifier_path="$HOME/models/imagenet/resnet50/noise_$sigma_str/checkpoint.pth.tar"
  dataset="imagenet"
elif [ "$option" = "c" ]; then
  base_classifier_path="$HOME/models/cifar10/resnet110/noise_$sigma_str/checkpoint.pth.tar"
  dataset="cifar10"
fi

output_file="$HOME/test_results/${dataset}_${sigma_str}.h5"
log_file="$HOME/test_results/${dataset}_${sigma_str}.log"

python -m certify.main_loop --base_classifier "$base_classifier_path" --sigma "$sigma" --outfile "$output_file" \
       --log_file "$log_file"

# Deactivate the virtual environment
deactivate
