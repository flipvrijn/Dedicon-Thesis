#!/bin/bash
#
#

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../downloads/" && pwd )"
cd $DIR

echo "Applying selective search to images..."

if [ $# -ne 2 ]; then
    echo "No split provided!"
    read -p "Split [train|val]: " split
    echo "No dataset provided!"
    read -p "Dataset [ilsvrc|coco]: " dataset
else
    split=$1
    dataset=$2
fi

selectivesearchfile=$DIR/selective_search_${dataset}_${split}.mat

python2.7 ../selective_search.py \
    -i $DIR/images/${dataset}_${split}/ \
    -o $selectivesearchfile