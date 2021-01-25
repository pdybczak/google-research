# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/bin/bash

# Main script for hyperparameter optimisation


# Modifiable experiment options.
# Expt options include {volatility, electricity, traffic, favorita}
EXPT=mpwik
OUTPUT_FOLDER=~/tft_outputs_distr  # Path to store data & experiment outputs
USE_GPU=yes

CUDA_VISIBLE_DEVICES=0, python3 -m script_distr_hyperparam_opt $EXPT $OUTPUT_FOLDER $USE_GPU yes $CUDA_VISIBLE_DEVICES 1 &
CUDA_VISIBLE_DEVICES=2, python3 -m script_distr_hyperparam_opt $EXPT $OUTPUT_FOLDER $USE_GPU yes $CUDA_VISIBLE_DEVICES 2 &
CUDA_VISIBLE_DEVICES=3, python3 -m script_distr_hyperparam_opt $EXPT $OUTPUT_FOLDER $USE_GPU yes $CUDA_VISIBLE_DEVICES 3 &
