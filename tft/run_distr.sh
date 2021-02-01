# Main script for hyperparameter optimisation


# Modifiable experiment options.
# Expt options include {volatility, electricity, traffic, favorita}
EXPT=mpwik
OUTPUT_FOLDER=~/tft_outputs_distr  # Path to store data & experiment outputs
USE_GPU=yes

set -e

CUDA_VISIBLE_DEVICES=0, python3 -m script_distr_hyperparam_opt $EXPT $OUTPUT_FOLDER $USE_GPU yes $CUDA_VISIBLE_DEVICES 1 &
CUDA_VISIBLE_DEVICES=2, python3 -m script_distr_hyperparam_opt $EXPT $OUTPUT_FOLDER $USE_GPU yes $CUDA_VISIBLE_DEVICES 2 &
CUDA_VISIBLE_DEVICES=3, python3 -m script_distr_hyperparam_opt $EXPT $OUTPUT_FOLDER $USE_GPU yes $CUDA_VISIBLE_DEVICES 3 &
