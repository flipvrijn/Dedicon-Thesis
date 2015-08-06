#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

outputdir=$DIR/output/_temp

if [ ! -d "$outputdir" ]; then
    mkdir $outputdir
fi

rcnn_out_file=$outputdir/rcnn_out
mlp_out_file=$outputdir/mlp_out

# Call regional convolutional neural network
# and store results of each region in file
python2.7 frcnn.py -i $1 \
                   -o ${rcnn_out_file}

# Call multilayer perceptron with output of
# regional convolutional neural network
python2.7 networks/mlp.py ${rcnn_out_file}.mat ${mlp_out_file}