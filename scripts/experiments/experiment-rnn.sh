DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../models/attention/" && pwd )"
cd $DIR

DATA_DIR="/media/Data/flipvanrijn/datasets/coco/processed/reduced/rnn/"
OUT_DIR="/media/Data/flipvanrijn/models/"
MODEL_NAME="context_attn_rnn.npz"

python2.7 eval.py --attn_type=deterministic --type=t_attn --lenc --tex_dim=500 $DATA_DIR $OUT_DIR $MODEL_NAME