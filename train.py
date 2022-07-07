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

import math
import os
import tensorflow as tf

import argparse
import checkpoints
import dataloader
import losses
import ssd300
import ssd_constants


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, steps_per_epoch, **kwargs):
        self.base_lr = base_lr
        self.steps_per_epoch = steps_per_epoch
        super(LRSchedule, self).__init__(**kwargs)

    @tf.function
    def __call__(self, step):
        epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
        warmup_epochs = 6
        sin = tf.sin(math.pi / 2.0 * epoch / warmup_epochs)
        cos = tf.cos(math.pi * tf.minimum((epoch - warmup_epochs) / 48.0, 1.0))
        return self.base_lr * (tf.where(epoch < warmup_epochs, sin, cos * 0.5 + 0.5) + 0.001)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "steps_per_epoch": self.steps_per_epoch
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', default="../../ssd_init.h5",
                        help='path to h5 with initial weights file')
    parser.add_argument('--data_dir', default="/data/coco2017/ssd_tf_records",
                        help='path to dataset')
    parser.add_argument('--model_dir', default="/tmp/ssd",
                        help='path to model_dir')
    parser.add_argument('-b', '--batch_size', metavar='N',
                        default=128, help='Batch size', type=int)
    parser.add_argument('-e', '--epochs', metavar='E',
                        default=64, help='Amount of epochs to train', type=int)
    parser.add_argument('--base_lr', default=3e-3, metavar='BASE_LR',
                        help='base learning rate (scaled wrt bs=32)', type=float)
    parser.add_argument('--wd', default=1e-3, metavar='WD',
                        help='weight decay', type=float)
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--num_examples', metavar='N', default=117266,
                        help='Amount of examples in the dataset', type=int)
    args = parser.parse_args()

    tf.config.optimizer.set_jit("autoclustering")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    params = {
        'batch_size': args.batch_size,
        'global_batch_size': args.batch_size,
        'num_shards': 1,
        'shard_index': 0,
        'dtype': tf.float32
    }
    callbacks = []

    try:
        from habana_frameworks.tensorflow import load_habana_module
        load_habana_module()
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
        params['dtype'] = tf.bfloat16
    except ImportError:
        pass

    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        import horovod.tensorflow.keras as hvd
        hvd.init()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
        params['global_batch_size'] *= hvd.size()
        params['shard_index'] = hvd.rank()
        params['num_shards'] = hvd.size()
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

    params['steps_per_epoch'] = args.num_examples // params['global_batch_size']

    ds = dataloader.SSDInputReader(
        args.data_dir + "/train-*", is_training=True)(params)
    image = tf.keras.layers.Input(
        (300, 300, 3), name="image", batch_size=args.batch_size, dtype=params['dtype'])
    model = ssd300.build(image)

    if params['shard_index'] == 0:
        model.summary()
        if not os.path.exists(args.init):
            args.init = checkpoints.get_resnet34_h5_checkpoint()
        model.load_weights(args.init)

    model.weight_decay = args.wd
    lr = LRSchedule(
        args.base_lr * params['global_batch_size'] / 32.0, params['steps_per_epoch'])
    opt = tf.keras.optimizers.SGD(
        learning_rate=lr, momentum=ssd_constants.MOMENTUM)

    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        with tf.keras.utils.custom_object_scope({"LRSchedule": LRSchedule}):
            opt = hvd.DistributedOptimizer(opt)

    model.compile(optimizer=opt, loss={"box": losses.localization_loss,
                  "cls": losses.classification_loss}, steps_per_execution=params['steps_per_epoch'])

    if params['shard_index'] == 0:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=args.model_dir, profile_batch=(20, 24) if args.profile else 0))

    model.fit(x=ds, epochs=args.epochs, steps_per_epoch=params['steps_per_epoch'],
              callbacks=callbacks, verbose=2 if params['shard_index'] == 0 else 0)

    if params['shard_index'] == 0:
        path = args.model_dir + "/ckpt-{:03d}.h5".format(args.epochs)
        print("Saving weights to ", path)
        model.save_weights(path)
