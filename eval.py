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

import argparse
import glob
import tensorflow as tf
from pycocotools.coco import COCO

import dataloader
import ssd300
from coco_metric import compute_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', metavar='N',
                        default=125, help='Batch size', type=int)
    parser.add_argument('--model_dir', default="/tmp/ssd",
                        help='path to model_dir')
    parser.add_argument('--data_dir', default="/data/coco2017/ssd_tf_records",
                        help='path to dataset')
    args = parser.parse_args()

    dtype = tf.float32
    try:
        from habana_frameworks.tensorflow import load_habana_module
        load_habana_module()
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
        dtype = tf.bfloat16
    except ImportError:
        pass

    batch_size = args.batch_size
    val_ds = dataloader.SSDInputReader(args.data_dir + "/val-*", is_training=False)({
        'batch_size': batch_size,
        'num_shards': 1,
        'shard_index': 0,
        'dtype': dtype,
        'eval_samples': 5000})

    image = tf.keras.layers.Input(
        (300, 300, 3), name="image", batch_size=batch_size, dtype=dtype)
    source_id = tf.keras.layers.Input(
        (), name="source_id", batch_size=batch_size, dtype=tf.int32)
    raw_shape = tf.keras.layers.Input(
        (3), name="raw_shape", batch_size=batch_size, dtype=tf.int32)

    model = ssd300.build(image, is_training=False,
                         raw_shape=raw_shape, source_id=source_id)

    coco_gt = COCO(args.data_dir +
                   "/raw-data/annotations/instances_val2017.json")

    writer = tf.summary.create_file_writer(args.model_dir + "/eval")

    with writer.as_default():
        ckpts = glob.glob(args.model_dir + "/*.h5")
        ckpts.sort()
        for ckpt in ckpts:
            print(ckpt)
            pos = ckpt.rfind('-')
            if pos == -1:
                continue
            epoch = int(ckpt[pos+1:-3])
            model.load_weights(ckpt)
            out = model.predict(val_ds)
            result = compute_map(out, coco_gt, use_cpp_extension=False)
            for key in result:
                print("Result epoch {} {} = {:.4f}".format(
                    epoch, key, result[key]))
                tf.summary.scalar(key, result[key], step=epoch)
                writer.flush()
