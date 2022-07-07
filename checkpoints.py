# Copyright 2022 Michal Szutenberg
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

import numpy as np
import pickle
import random
import requests
import string
import tensorflow as tf
import torch

import ssd300


def get_resnet34_ckpt_pth():
    rand = random.choices(string.ascii_lowercase, k=10)
    path = "/tmp/" + ''.join(rand) + "_resnet34--333f7ec4.pth"
    URL = "https://download.pytorch.org/models/resnet34-333f7ec4.pth"
    print("Downloading {} to {}".format(URL, path))
    response = requests.get(URL)
    open(path, "wb").write(response.content)
    return path


def convert_pth_to_pickle(file):
    path = file + ".pickle"
    print("Converting {} to {}...".format(file, path))
    pth_input = torch.load(open(file, 'rb'))
    out = {}
    for key, value in pth_input.items():
        out[key] = value.data.numpy()
    pickle.dump(out, open(path, 'wb'))
    return path


def convert_pickle_to_h5(file):
    path = file + ".h5"
    var_mapping = {
        "kernel": "weight",
        "beta": "bias",
        "gamma": "weight",
        "moving_mean": "running_mean",
        "moving_variance": "running_var"
    }

    image = tf.keras.layers.Input((300, 300, 3))
    model = ssd300.build(image)

    with open(file, 'rb') as pickle_file:
        content = pickle.load(pickle_file)

    for layer in model.layers:
        for weight in layer.weights:
            try:
                pos = weight.name.rfind('/')
                layer = weight.name[:pos]
                var = var_mapping[weight.name[pos+1:-2]]
                pickle_name = layer + "." + var
                value = content[pickle_name]
            except KeyError:
                print("Skipping {:<80} {}".format(weight.name, weight.shape))
                continue
            if "kernel" in weight.name:
                value = np.transpose(value, (3, 2, 1, 0))
            print("Saving   {:<40} => {:<36} {}".format(
                weight.name, pickle_name, value.shape))
            tf.keras.backend.set_value(weight, value)
    print("Saving weights to " + path)
    model.save_weights(path)
    return path


def get_resnet34_h5_checkpoint():
    f1 = get_resnet34_ckpt_pth()
    f2 = convert_pth_to_pickle(f1)
    return convert_pickle_to_h5(f2)
