# ssd300
Single Shot MultiBox Detector with 300x300 input and ResNet34 backbone, TensorFlow 2

## Introduction
This repository contains SSD model based on [SSD mlperf 0.6](https://github.com/mlcommons/training_results_v0.6.git) script.

## Main differences

* Dataloader was simplified
* Custom LRSchedule (cosine decay based)
* First convolution and batch normalization in RN34 backbone are frozen
* Added Horovod support
* Pure TensorFlow 2 code

## HowTo

To be done...

## Results

mAP above 0.23 after 50 epochs on 8 cards, bfloat16, batch size 128 each, default params are used.
