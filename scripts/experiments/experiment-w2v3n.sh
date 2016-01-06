DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../models/attention/" && pwd )"
cd $DIR

DATA_DIR="/media/Data/flipvanrijn/datasets/coco/processed/reduced/w2v3ngram/"
OUT_DIR="/media/Data/flipvanrijn/models/"
MODEL_NAME="context_attn_w2v3ngram.npz"

python2.7 eval.py --attn_type=deterministic --type=t_attn $DATA_DIR $OUT_DIR $MODEL_NAME