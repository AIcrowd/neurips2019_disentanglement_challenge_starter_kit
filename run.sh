#!/bin/bash

source ./train_environ.sh

# If you have other dependencies, this would be a nice place to
# add them to your PYTHONPATH:
#export PYTHONPATH=${PYTHONPATH}:path/to/your/dependency

# Comment and uncomment as required:
# Tensorflow
#export PYTHONPATH=${PYTHONPATH}:./tensorflow
#python ./tensorflow/train_tensorflow.py

# Pytorch
#export PYTHONPATH=${PYTHONPATH}:./pytorch
python ./pytorch/train_pytorch.py --epochs 1

# Numpy
#export PYTHONPATH=${PYTHONPATH}:./numpy
#python ./numpy/train_numpy.py
