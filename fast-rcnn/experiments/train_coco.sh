#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/`basename $0`.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/solver-coco.prototxt \
  --cfg experiments/coco.yml \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb coco_train2014 \
  --iters 640000
# there are 40000 iterations for voc_2007_trainval, which contains 5000 images
# voc_2007_trainval contains 40000 images, there are 8 times more images in coco_val2014 than in voc_2007_trainval, so using 320K iterations for coco_val2014
# there are 2 folds images in coco_train2014, so should be 640K iterations

exit

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/test.prototxt \
  --net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000.caffemodel \
  --imdb coco_val2014
