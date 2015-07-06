#!/bin/bash

# Processes extra context information for a given dataset.
# 
# Usage: ./preprocess.sh <key> <secret> <train|val>
#

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../downloads/" && pwd )"
cd $DIR

# If less then 3 arguments, then ask for input
if [ $# -ne 3 ]; then
  echo "Please provide the following information: "
  read -p "Flickr API key: " key
  read -p "Flickr API secret: " secret
  read -p "Split [train|val]: " split
else
  key=$1
  secret=$2
  split=$3
fi

instancesfile=$DIR/instances_${split}2014.json
contextfile=$DIR/coco_${split}_context.json
sanitizedfile=$DIR/coco_${split}_sanitized.json
datasetfile=$DIR/coco_${split}_dataset.json

if [ -f $contextfile ]; then
  echo "Skipping fetching Flickr context..."
else
  echo '>>> Adding Flickr context to dataset'
  python2.7 ../coco/coco_context.py \
    --key $key \
    --secret $secret \
    --instances $instancesfile \
    -o $contextfile \
    --cp $DIR
fi

if [ -f $sanitizedfile ]; then
  echo "Skipping sanitizing Flickr context..."
else
  echo '>>> Sanitizing Flickr context dataset'
  python2.7 ../coco/coco_sanitize.py \
    -i $contextfile \
    -o $sanitizedfile \
    --cp $DIR
fi

if [ -f $datasetfile ]; then
  echo "Skipping building dataset file..."
else
  echo ">>> Building dataset file"
  python2.7 ../coco/coco_dataset.py \
    --context=$contextfile \
    --caption=$sanitizedfile \ 
    --instances=$instancesfile \ 
    --cp=$DIR \
    -o=$datasetfile
fi

echo "Done!"