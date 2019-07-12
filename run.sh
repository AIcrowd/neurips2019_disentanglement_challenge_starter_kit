#!/bin/bash

# Root is where this file is.
export NDC_ROOT="$( cd "$(dirname "$0")" ; pwd -P )"

# Source the training environment (see the env variables defined therein) if we are not evaluating
if [ ! -n "${AICROWD_IS_GRADING+set}" ]; then
  # AICROWD_IS_GRADING is not set, so we're not running on the evaluator and it's safe to
  # source the train_environ.sh
  source ${NDC_ROOT}/train_environ.sh
else
  # We're on the evaluator.
  # Add root to python path, since this would usually be done in train_environ.sh
  export PYTHONPATH=${PYTHONPATH}:${NDC_ROOT}
fi

# If you have other dependencies, this would be a nice place to
# add them to your PYTHONPATH:
#export PYTHONPATH=${PYTHONPATH}:path/to/your/dependency

# Comment and uncomment as required...
# Tensorflow:
#export PYTHONPATH=${PYTHONPATH}:${NDC_ROOT}/tensorflow
#python ${NDC_ROOT}/tensorflow/train_tensorflow.py

# Pytorch:
export PYTHONPATH=${PYTHONPATH}:${NDC_ROOT}/pytorch
python ${NDC_ROOT}/pytorch/train_pytorch.py --epochs 1

# Numpy:
#export PYTHONPATH=${PYTHONPATH}:${NDC_ROOT}/numpy
#python ${NDC_ROOT}/numpy/train_numpy.py

# Execute the local evaluation
#echo "----- LOCAL EVALUATION -----"
#python ${NDC_ROOT}/local_evaluation.py