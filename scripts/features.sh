#!/bin/bash
#
#

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../downloads/" && pwd )"
cd $DIR

echo "Applying selective search to images..."

if [ $# -ne 1 ]; then
    echo "No split provided!"
    read -p "Split [train|val]: " split
else
    split=$1
fi

selectivesearchfile=$DIR/selective_search_${split}.mat

python2.7 ../selective_search.py \
    -i $DIR/images/${split}/ \
    -o $selectivesearchfile