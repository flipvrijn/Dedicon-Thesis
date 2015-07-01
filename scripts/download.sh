#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../downloads/" && pwd )"
cd $DIR

websites=(
  http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/fast_rcnn_models.tgz
  http://msvocds.blob.core.windows.net/annotations-1-0-2/instances_train2014.json
  http://msvocds.blob.core.windows.net/annotations-1-0-2/instances_val2014.json
  http://msvocds.blob.core.windows.net/annotations-1-0-2/captions_train2014.json
  http://msvocds.blob.core.windows.net/annotations-1-0-2/captions_val2014.json
)
files=(
  fast_rcnn_models.tgz
  instances_train2014.json
  instances_val2014.json
  captions_train2014.json
  captions_val2014.json
)
checksums=(
  5f7dde9f5376e18c8e065338cc5df3f7
  0dcea1764a9efa9d764738631d8006f3
  da37ab3e12c57ce045f354e0a990316a
  6329188a179c47223bf21bcd174ee991
  e27594e83f3752905d7093d1b3d57aef
)

# Verifies the correctness of the MD5 checksum of a file.
# param: $1 is filename
# param: $2 is the required checksum
# Returns: boolean
function check {
  local result=0
  if [ -f $1 ]; then
    os=`uname -s`
    if [ "$os" = "Linux" ]; then
      checksum=`md5sum $1 | awk '{ print $1 }'`
    elif [ "$os" = "Darwin" ]; then
      checksum=`cat $1 | md5`
    fi
    if [ "$checksum" = "$2" ]; then
      local result=1
    else
      local result=0
    fi
  fi
  echo "$result"
}

# Loops through the files and will download them if the checksum
# is not correct or the file does not exist. 
# Will ask the user if the file should be downloaded.
for index in ${!files[*]}
do
  read -p "Download ${files[$index]}? [y/n] " -n 1 -r
  echo 
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    url=${websites[$index]}
    result=$(check ${files[$index]} ${checksums[$index]})
    if [ "$result" = "1" ]; then
      echo "Skipping ${files[$index]}."
    else
      echo "Downloading ${files[$index]}..."

      wget $url -nv --show-progress -O ${files[$index]}
      
      tar zxvf ${files[$index]}
    fi
  else
    echo "Skipping ${files[$index]}."
  fi
done

echo "Done!"