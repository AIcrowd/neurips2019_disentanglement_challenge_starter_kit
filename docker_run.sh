#!/bin/bash


if [ -e environ_secret.sh ]
then
    echo "Note: Gathering environment variables from environ_secret.sh"
    source environ_secret.sh
else
    echo "Note: Gathering environment variables from environ.sh"
    source environ.sh
fi

# Expected Env variables : in environ.sh

# Clean up shared directory
sudo rm -rf $OUTPUT_DIRECTORY/*

sudo nvidia-docker run \
    --net=host \
    --privileged \
    --user 0 \
    -v `pwd`/$DATASET:/DATASET \
    -v `pwd`/$OUTPUT_DIRECTORY:/SHARED \
    -v `pwd`:/www \
    --workdir /www \
    -e DISENTANGLEMENT_LIB_DATA="/DATASET" \
    -e AICROWD_OUTPUT_PATH="/SHARED" \
    -e AICROWD_EVALUATION_NAME="experiment_name" \
    -e AICROWD_DATASET_NAME="cars3d" \
    -it ${IMAGE_NAME}:${IMAGE_TAG} \
    /www/run.sh

# /home/aicrowd/run_wrapper.sh
# Note: Running in Privileged mode during local debugs
# This helps with permission issues on the mounted volumes