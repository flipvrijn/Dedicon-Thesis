#!/bin/bash
#
#

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../downloads/images/" && pwd )"
cd $DIR

echo "Applying selective search to images..."

if [ $# -ne 1 ]; then
    echo "No split provided!"
    read -p "Split [train|val]: " split
else
    split=$1

python2.7 ../selective_search.py \
    -i $DIR/$split \
    -o $DIR/../boxes_${split}.mat