# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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


# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import tensorflow as tf
from numba import cuda
import gin.tf
import aicrowd_helpers

# 0. Settings
# ------------------------------------------------------------------------------
# By default, we save all the results in subdirectories of the following path.
base_path = os.getenv("AICROWD_OUTPUT_PATH", "../scratch/shared")
experiment_name = os.getenv("AICROWD_EVALUATION_NAME", "experiment_name")
DATASET_NAME = os.getenv("AICROWD_DATASET_NAME", "cars3d")
ROOT = os.getenv("NDC_ROOT", "..")
overwrite = True

# 0.1 Helpers
# ------------------------------------------------------------------------------


def get_full_path(filename):
    return os.path.join(ROOT, "tensorflow", filename)


########################################################################
# Register Execution Start
########################################################################
aicrowd_helpers.execution_start()


# Train a custom VAE model.
@gin.configurable("BottleneckVAE") 
class BottleneckVAE(vae.BaseVAE):
  """BottleneckVAE.

  The loss of this VAE-style model is given by:
    loss = reconstruction loss + gamma * |KL(app. posterior | prior) - target|
  """

  def __init__(self, gamma=gin.REQUIRED, target=gin.REQUIRED):
    self.gamma = gamma
    self.target = target

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    # This is how we customize BaseVAE. To learn more, have a look at the
    # different models in 
    # https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/unsupervised/vae.py
    del z_mean, z_logvar, z_sampled
    return self.gamma * tf.math.abs(kl_loss - self.target)

gin_bindings = [
    "dataset.name = '{}'".format(DATASET_NAME),
    "model.model = @BottleneckVAE()",
    "BottleneckVAE.gamma = 4",
    "BottleneckVAE.target = 10."
]
# Call training module to train the custom model.
experiment_output_path = os.path.join(base_path, experiment_name)

########################################################################
# Register Progress (start of training)
########################################################################
aicrowd_helpers.register_progress(0.0)

train.train_with_gin(
    os.path.join(experiment_output_path, "model"), overwrite,
    [get_full_path("model.gin")], gin_bindings)

########################################################################
# Register Progress (end of training, start of representation extraction)
########################################################################
aicrowd_helpers.register_progress(0.90)

# Extract the mean representation for both of these models.
representation_path = os.path.join(experiment_output_path, "representation")
model_path = os.path.join(experiment_output_path, "model")
# This contains the settings:
postprocess_gin = [get_full_path("postprocess.gin")]
postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                 postprocess_gin)

print("Written output to : ", experiment_output_path)
########################################################################
# Register Progress (of representation extraction)
########################################################################
aicrowd_helpers.register_progress(1.0)

########################################################################
# Submit Results for evaluation
########################################################################
cuda.close() 
aicrowd_helpers.submit()
