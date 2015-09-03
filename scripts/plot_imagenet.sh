DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../utils/" && pwd )"
cd $DIR

LOGSDIR="/home/flipvanrijn/Workspace/Dedicon-Thesis/networks/fast-rcnn/experiments/logs/"
python2.7 loss_plot.py -i $LOGSDIR"$(ls ${LOGSDIR} | tail -1)"