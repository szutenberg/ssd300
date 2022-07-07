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

import dataloader
import ssd_constants
from utils.object_detection import box_coder
from utils.object_detection import box_list
from utils.object_detection import faster_rcnn_box_coder


class ConcatBox(tf.keras.layers.Layer):
    def call(self, box):
        flat_box = []
        it = zip(ssd_constants.FEATURE_SIZES, ssd_constants.NUM_DEFAULTS)
        for i, (scale, num_default) in enumerate(it):
            shape = [-1, scale ** 2 * num_default, 4]
            flat_box.append(tf.reshape(box[i], shape))
        return tf.concat(flat_box, axis=1)


class ConcatCls(tf.keras.layers.Layer):
    def call(self, cls):
        flat_cls = []
        it = zip(ssd_constants.FEATURE_SIZES, ssd_constants.NUM_DEFAULTS)
        for i, (scale, num_default) in enumerate(it):
            shape = [-1, scale ** 2 * num_default, ssd_constants.NUM_CLASSES]
            flat_cls.append(tf.reshape(cls[i], shape))
        return tf.concat(flat_cls, axis=1)


class Predictions(tf.keras.layers.Layer):
    def call(self, inputs):
        flattened_box = inputs['box_out']
        flattened_cls = inputs['cls_out']
        ssd_box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
            scale_factors=ssd_constants.BOX_CODER_SCALES)

        anchors = box_list.BoxList(
            tf.convert_to_tensor(dataloader.DefaultBoxes()('ltrb')))

        decoded_boxes = box_coder.batch_decode(
            encoded_boxes=flattened_box, box_coder=ssd_box_coder, anchors=anchors)

        # [bs, SSD_boxes, N_classes]
        pred_scores = tf.nn.softmax(flattened_cls, axis=2)

        top_k_scores, top_k_indices = tf.nn.top_k(
            tf.transpose(pred_scores, [0, 2, 1]), k=200, sorted=True)
        pred_scores = tf.transpose(top_k_scores, [0, 2, 1])
        indices = tf.transpose(top_k_indices, [0, 2, 1])

        return decoded_boxes, pred_scores, indices
