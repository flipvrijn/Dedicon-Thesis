#!/bin/bash

# Processes extra context information for a given dataset.
# 
# Usage: ./preprocess.sh <key> <secret> <train|val>
#

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../networks/fast-rcnn/" && pwd )"
cd $DIR

sh ./experiments/train_coco.sh 0