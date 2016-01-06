DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../models/attention/" && pwd )"
cd $DIR

DATA_DIR="/media/Data/flipvanrijn/datasets/coco/processed/full/"
OUT_DIR="/media/Data/flipvanrijn/models/"
MODEL_NAME="baseline.npz"

python2.7 eval.py --attn_type=deterministic --type=normal $DATA_DIR $OUT_DIR $MODEL_NAME
