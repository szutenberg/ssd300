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

import tensorflow as tf
import re

import layers
import ssd_constants

conv2d = tf.keras.layers.Conv2D


def bn(name, trainable=True):
    def layer(input):
        return tf.keras.layers.BatchNormalization(epsilon=1.0e-5, name=name, trainable=trainable)(input)
    return layer


def residual_block(filters, use_1x1conv=False, strides=1, name="", trainable=True):
    def layer(input_tensor):
        if use_1x1conv:
            conv0 = conv2d(filters, 1, strides=strides, padding="same",
                           use_bias=False, trainable=trainable, name=name+".downsample.0")(input_tensor)
            shortcut = bn(name+".downsample.1", trainable=trainable)(conv0)
        else:
            shortcut = input_tensor
        conv1 = conv2d(filters, 3, strides=strides, padding="same",
                       use_bias=False, name=name+".conv1", trainable=trainable)(input_tensor)
        bn1 = bn(name+".bn1")(conv1)
        relu1 = tf.keras.layers.ReLU(name=name+".bn1.relu")(bn1)
        conv2 = conv2d(filters, 3, padding="same",
                       use_bias=False, name=name+".conv2", trainable=trainable)(relu1)
        bn2 = bn(name+".bn2", trainable=trainable)(conv2)
        add = tf.keras.layers.Add(name=name + ".add")([bn2, shortcut])
        out = tf.keras.layers.ReLU(name=name)(add)
        return out
    return layer


class SSD300(tf.keras.Model):
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs)
        self.weight_decay = None

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
            variables_wd = []
            for v in self.trainable_variables:
                if "bn" in v.name:
                    continue
                variables_wd.append(v)
            if self.weight_decay:
                loss += self.weight_decay * \
                    tf.add_n([tf.nn.l2_loss(v) for v in variables_wd])

        if "horovod" in str(type(self.optimizer)):
            import horovod.tensorflow as hvd
            tape = hvd.DistributedGradientTape(tape)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        return self.compute_metrics(x, y, y_pred, None)


def build(image, is_training=True, raw_shape=None, source_id=None):
    x = conv2d(64, 7, strides=(2, 2), padding="same",
               use_bias=False, name="conv1", trainable=False)(image)
    x = bn("bn1", trainable=False)(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(3, 3), strides=2, padding="same")(x)

    for layer, (block_cnt, channels) in enumerate(zip([0, 3, 4, 6], [0, 64, 128, 256])):
        for i in range(block_cnt):
            name = "layer" + str(layer) + "." + str(i)
            x = residual_block(channels,
                               use_1x1conv=(layer > 1) and (i == 0),
                               strides=2 if (layer == 2 and i == 0) else 1,
                               name=name, trainable=True
                               )(x)

    feats = {}
    feats[3] = x
    feats[4] = conv2d(256, 1, padding="same",
                      activation="relu", name="block7-conv1x1")(x)
    feats[4] = conv2d(512, 3, strides=2, padding="same",
                      activation="relu", name="block7-conv3x3")(feats[4])
    feats[5] = conv2d(256, 1, padding="same", activation="relu",
                      name="block8-conv1x1")(feats[4])
    feats[5] = conv2d(512, 3, strides=2, padding="same",
                      activation="relu", name="block8-conv3x3")(feats[5])
    feats[6] = conv2d(128, 1, padding="same", activation="relu",
                      name="block9-conv1x1")(feats[5])
    feats[6] = conv2d(256, 3, strides=2, padding="same",
                      activation="relu", name="block9-conv3x3")(feats[6])
    feats[7] = conv2d(128, 1, padding="same", activation="relu",
                      name="block10-conv1x1")(feats[6])
    feats[7] = conv2d(256, 3, padding="valid", activation="relu",
                      name="block10-conv3x3")(feats[7])
    feats[8] = conv2d(128, 1, padding="same", activation="relu",
                      name="block11-conv1x1")(feats[7])
    feats[8] = conv2d(256, 3, padding="valid", activation="relu",
                      name="block11-conv3x3")(feats[8])

    cls_outs = []
    box_outs = []
    for i, num_defaults in ssd_constants.NUM_DEFAULTS_BY_LEVEL.items():
        cls_outs.append(conv2d(ssd_constants.NUM_CLASSES * num_defaults, 3,
                        padding="same", name="cls-" + str(i))(feats[i]))
        box_outs.append(conv2d(4 * num_defaults, 3,
                        padding="same", name="box-" + str(i))(feats[i]))

    cls_out = layers.ConcatCls()(cls_outs)
    cls_out = tf.keras.layers.Activation(
        tf.identity, dtype=tf.float32, name="cls")(cls_out)
    box_out = layers.ConcatBox()(box_outs)
    box_out = tf.keras.layers.Activation(
        tf.identity, dtype=tf.float32, name="box")(box_out)

    if is_training:
        model = SSD300(inputs=image, outputs=[box_out, cls_out])
    else:
        decoded_boxes, pred_scores, indices = layers.Predictions(
            dtype=tf.float32)({'box_out': box_out, 'cls_out': cls_out})
        decoded_boxes = tf.keras.layers.Activation(
            tf.identity, dtype=tf.float32, name="decoded_boxes")(decoded_boxes)
        pred_scores = tf.keras.layers.Activation(
            tf.identity, dtype=tf.float32, name="pred_scores")(pred_scores)
        indices = tf.keras.layers.Activation(
            tf.identity, dtype=tf.float32, name="indices")(indices)
        model = SSD300(inputs=[image, source_id, raw_shape], outputs=[
            decoded_boxes, pred_scores, indices, raw_shape, source_id])

    return model
