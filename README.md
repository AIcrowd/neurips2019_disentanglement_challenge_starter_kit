![AIcrowd-Logo](https://raw.githubusercontent.com/AIcrowd/AIcrowd/master/app/assets/images/misc/aicrowd-horizontal.png)

# NeurIPS 2019 : Disentanglement Challenge Starter Kit

[![gitter-badge](https://badges.gitter.im/AIcrowd-HQ/disentanglement_challenge.png)](http://gitter.im/AIcrowd-HQ/disentanglement_challenge)

Instructions to make submissions to the [NeurIPS 2019 : Disentanglement Challenge](https://www.aicrowd.com/challenges/neurips-2019-disentanglement-challenge).

Participants will have to submit their code, with packaging specifications, and the evaluator will automatically build a docker image and execute their code against a series of secret datasets.

![](https://i.imgur.com/zr8RUr1.gif)

### Setup

- **docker** : By following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- **nvidia-docker** : By following the instructions [here](<https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)>)
- **aicrowd-repo2docker** [This [package](https://pypi.org/project/aicrowd-repo2docker/) requires python 3.4+]

```sh
pip install -U aicrowd-repo2docker
```

- **Anaconda** (By following instructions [here](https://www.anaconda.com/download)) At least version `4.5.11` is required to correctly populate `environment.yml`.

* **Your code specific dependencies**

```sh
# If say you want to install PyTorch
conda install pytorch torchvision -c pytorch
```

### Clone repository

```
git clone git@github.com:AIcrowd/neurips2019_disentanglement_challenge_starter_kit.git
cd neurips2019_disentanglement_challenge_starter_kit
pip install -r requirements.txt
```

### Download and Prepare Dataset

Follow the instructions [here](https://github.com/google-research/disentanglement_lib#downloading-the-data-sets) to download the publicly available datasets, and store them inside `./scratch/datasets` folder. Please ensure that the datasets are not checked into the repository.

### Test Submission Locally

#### Note: This is just a test submission which will only train for 10 steps (to keep training time short). To attain reasonable performance, about 300'000 training steps are recommended. This setting can be changed in model.gin.

```
cd neurips2019_disentanglement_challenge_starter_kit

# Build docker image for your submission
./build.sh

##
## You can update any custom env variables in `environ.sh`

# In a separate tab :
# Finally, run your agent locally by :
./docker_run.sh

# Now you should see the output of your training inside the
# ./scratch/shared folder

# Then you can do a simple local evaluation by running :
export DISENTANGLEMENT_LIB_DATA="./scratch/dataset/"
python local_evaluation.py
```

### Train and Test locally with Pytorch
```
cd neurips2019_disentanglement_challenge_starter_kit

# Make sure that your datasets live in "./scatch/dataset". 
# If they do not, change train_environ.sh accordingly. 
source train_environ.sh
export AICROWD_EVALUATION_NAME=myvae

# Train a plain old VAE with Pytorch. 
# See train_pytorch.py for more instructions
python train_pytorch.py --epochs 10

# Run the local evaluation
python local_evaluation.py
```

# How do I specify my software runtime ?

The software runtime is specified by exporting your `conda` env to the root
of your repository by doing :

```
conda env export --no-build > environment.yml
```

This `environment.yml` file will be used to recreate the `conda environment` inside the Docker container.
This repository includes an example `environment.yml`

# What should my code structure be like ?

Please follow the structure documented in the included [run.py](https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit/blob/master/run.py) to adapt
your already existing code to the required structure for this round.

## Important Concepts

### Repository Structure

- `aicrowd.json`
  Each repository should have a `aicrowd.json` with the following content :

```json
{
  "challenge_id": "aicrowd-neurips-2019-disentanglement-challenge",
  "grader_id": "aicrowd-neurips-2019-disentanglement-challenge",
  "authors": ["your-aicrowd-username"],
  "description": "sample description about your awesome agent",
  "license": "MIT",
  "gpu": true
}
```

This is used to map your submission to the said challenge, so please remember to use the correct `challenge_id` and `grader_id` as specified above.

Please specify if your code will a GPU or not for the evaluation of your model. If you specify `true` for the GPU, a **NVIDIA Tesla K80 GPU** will be provided and used for the evaluation.

### Packaging of your software environment

You can specify your software environment by using all the [available configuration options of repo2docker](https://repo2docker.readthedocs.io/en/latest/config_files.html). (But please remember to use [aicrowd-repo2docker](https://pypi.org/project/aicrowd-repo2docker/) to have GPU support)

The recommended way is to use Anaconda configuration files using **environment.yml** files.

```sh
# The included environment.yml is generated by the command below, and you do not need to run it again
# if you did not add any custom dependencies

conda env export --no-build > environment.yml

# Note the `--no-build` flag, which is important if you want your anaconda env to be replicable across all
# Or if you have a rather simple softwar runtime, then even a :
#
# pip freeze > requirements.txt
#
# could work.
```

### Debugging the packaged software environment

If you have issues with your submission because of your software environment and dependencies, you can debug them, by first building the docker image, and then getting a shell inside the image by :

```
nvidia-docker run --net=host -it $IMAGE_NAME /bin/bash
```

and then exploring to find the cause of the issue.

### Code Entrypoint

The evaluator will use `/home/aicrowd/run.sh` as the entrypoint, so please remember to have a `run.sh` at the root, which can instantitate any necessary environment variables, and also start executing your actual code. This repository includes a sample `run.sh` file.
If you are using a Dockerfile to specify your software environment, please remember to create a `aicrowd` user, and place the entrypoint code at `run.sh`.

## Submission

To make a submission, you will have to create a private repository on [https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/).

You will have to add your SSH Keys to your GitLab account by following the instructions [here](https://docs.gitlab.com/ee/gitlab-basics/create-your-ssh-keys.html).
If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

Then you can create a submission by making a _tag push_ to your repository on [https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/).
**Any tag push (where the tag name begins with "submission-") to your private repository is considered as a submission**  
Then you can add the correct git remote, and finally submit by doing :

```
cd neurips2019_disentanglement_challenge_starter_kit
# Add AIcrowd git remote endpoint
git remote add aicrowd git@gitlab.aicrowd.com:<YOUR_AICROWD_USER_NAME>/neurips2019_disentanglement_challenge_starter_kit.git
git push aicrowd master

# Create a tag for your submission and push
git tag -am "submission-v0.1" submission-v0.1
git push aicrowd master
git push aicrowd submission-v0.1

# Note : If the contents of your repository (latest commit hash) does not change,
# then pushing a new tag will **not** trigger a new evaluation.
```

You now should be able to see the details of your submission at :
[gitlab.aicrowd.com/<YOUR_AICROWD_USER_NAME>/neurips2019_disentanglement_challenge_starter_kit/issues](gitlab.aicrowd.com//<YOUR_AICROWD_USER_NAME>/neurips2019_disentanglement_challenge_starter_kit/issues)

**NOTE**: Remember to update your username in the link above :wink:

In the link above, you should start seeing something like this take shape (each of the steps can take a bit of time, so please be patient too :wink: ) :
![](https://i.imgur.com/69nf0Te.png)

and if everything works out correctly, then you should be able to see the final scores like this :
![](https://i.imgur.com/FPIqPKb.png)

**Best of Luck** :tada: :tada:

# Author

Sharada Mohanty <https://twitter.com/MeMohanty>
