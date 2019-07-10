#!/usr/bin/env bash

# Check if the root is set; if not use the location of this script as root
if ![ -n "${NDC_ROOT+set}" ]; then
  export NDC_ROOT="$( cd "$(dirname "$0")" ; pwd -P )"
fi

export PYTHONPATH=${PYTHONPATH}:${NDC_ROOT}

# Set up training environment.
# Feel free to change these as required:
export AICROWD_EVALUATION_NAME=experiment_name
export AICROWD_DATASET_NAME=cars3d

# Change these only if you know what you're doing:
export AICROWD_OUTPUT_PATH=${NDC_ROOT}/scratch/shared
export DISENTANGLEMENT_LIB_DATA=${NDC_ROOT}/scratch/dataset
