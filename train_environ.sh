#!/usr/bin/env bash

# Set up training environment. You might need to set the evaluation name and
# the dataset manually.
export AICROWD_OUTPUT_PATH=./scratch/shared
export AICROWD_EVALUATION_NAME=experiment_name
export AICROWD_DATASET_NAME=cars3d
export DISENTANGLEMENT_LIB_DATA=./scratch/dataset