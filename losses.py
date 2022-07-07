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

import ssd_constants


def huber_loss(y_true, y_pred):
    d = 1
    x = tf.abs(y_true - y_pred)
    mask = tf.cast(x > d, tf.float32)
    return 0.5*x*x*(1-mask) + mask*(0.5*d*d + d*(x-d))


def get_gt(y_true):
    gt_boxes, gt_cls, num_matched_boxes = tf.split(
        y_true, [ssd_constants.NUM_SSD_BOXES*4, ssd_constants.NUM_SSD_BOXES, 1], axis=1)
    gt_boxes = tf.reshape(gt_boxes, [-1, ssd_constants.NUM_SSD_BOXES, 4])
    num_matched_boxes = tf.reshape(num_matched_boxes, [-1])
    return gt_boxes, gt_cls, num_matched_boxes


def localization_loss(y_true, y_pred):
    gt_boxes, gt_cls, num_matched_boxes = get_gt(y_true)
    loss = tf.reduce_sum(huber_loss(gt_boxes, y_pred),
                         axis=-1) * tf.cast(gt_cls > 0, tf.float32)
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1) / num_matched_boxes)


@tf.custom_gradient
def _softmax_cross_entropy(logits, label):
    """Helper function to compute softmax cross entropy loss."""
    shifted_logits = logits - tf.expand_dims(tf.reduce_max(logits, -1), -1)
    exp_shifted_logits = tf.math.exp(shifted_logits)
    sum_exp = tf.reduce_sum(exp_shifted_logits, -1)
    log_sum_exp = tf.math.log(sum_exp)
    one_hot_label = tf.one_hot(label, ssd_constants.NUM_CLASSES)
    shifted_logits = tf.reduce_sum(shifted_logits * one_hot_label, -1)
    loss = log_sum_exp - shifted_logits

    def grad(dy):
        return (exp_shifted_logits / tf.expand_dims(sum_exp, -1) -
                one_hot_label) * tf.expand_dims(dy, -1), dy

    return loss, grad


def classification_loss(y_true, y_pred):
    _, gt_cls, num_matched_boxes = get_gt(y_true)

    cross_entropy = _softmax_cross_entropy(y_pred, tf.cast(gt_cls, tf.int32))
    pos_mask = tf.cast(gt_cls > 0, tf.float32)

    # Hard example mining
    neg_cross_entropy = cross_entropy * (1 - pos_mask)
    num_neg_boxes = tf.expand_dims(3 * num_matched_boxes, -1)
    _, indices = tf.nn.top_k(neg_cross_entropy, ssd_constants.NUM_SSD_BOXES)
    _, neg_cross_entropy_rank = tf.nn.top_k(-1 *
                                            indices, ssd_constants.NUM_SSD_BOXES)
    neg_mask = tf.cast(num_neg_boxes > tf.cast(
        neg_cross_entropy_rank, tf.float32), tf.float32)

    loss = cross_entropy * (pos_mask + neg_mask)

    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1) / num_matched_boxes)
