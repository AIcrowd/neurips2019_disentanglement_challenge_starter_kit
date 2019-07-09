# Frequently Asked Questions

* **How can I build a docker image from this repository?**

```sh
pip install aicrowd-repo2docker
REPO2DOCKER="$(which aicrowd-repo2docker)"
sudo ${REPO2DOCKER} --no-run \
  --user-id 1001 \
  --user-name aicrowd \
  --image-name disentanglement_challenge_submission \
--debug .
```

* **How can I run the previously build docker image to simulate a submission?**

```sh
export DATASET=./scratch/dataset/
export OUTPUT_DIRECTORY=./scratch/shared/

# Clean up shared directory
sudo rm -rf $OUTPUT_DIRECTORY/*

sudo nvidia-docker run \
    --net=host \
    --privileged \
    --rm \
    --user 0 \
    -v `pwd`/$DATASET:/DATASET \
    -v `pwd`/$OUTPUT_DIRECTORY:/SHARED \
    -v `pwd`:/www \
    --workdir /www \
    -e DISENTANGLEMENT_LIB_DATA="/DATASET" \
    -e AICROWD_OUTPUT_PATH="/SHARED" \
    -e AICROWD_EVALUATION_NAME="experiment_name" \
    -e AICROWD_DATASET_NAME="mpi3d_toy" \
    -it ${IMAGE_NAME}:${IMAGE_TAG} \
    /www/run.sh

```
