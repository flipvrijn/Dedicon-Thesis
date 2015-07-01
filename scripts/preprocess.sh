#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../downloads/" && pwd )"
cd $DIR

instancesfile=$DIR/instances_train2014.json
contextfile=$DIR/flickr_data_train_context.json
sanitizedfile=$DIR/flickr_data_train_completed.json

if [ -f $contextfile ]; then
  echo "Skipping fetching Flickr context..."
else
  if [ $# -ne 2 ]; then
    echo "Please provide the following information: "
    read -p "Flickr API key: " key
    read -p "Flickr API secret: " secret
  else
    key=$1
    secret=$2
  fi

  echo '>>> Adding Flickr context to dataset'
  python2.7 ../coco/coco_context.py \
    --key $key \
    --secret $secret \
    --annotation $instancesfile \
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

echo "Done!"