DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../models/attention/" && pwd )"
cd $DIR

DATA_DIR="/media/Data/flipvanrijn/datasets/coco/processed/reduced/w2v1ngram/"
OUT_DIR="/media/Data/flipvanrijn/models/"
MODEL_NAME="context_attn_w2v1ngram.npz"

python2.7 eval.py --attn_type=deterministic --type=t_attn $DATA_DIR $OUT_DIR $MODEL_NAME

# Generate caps
OUT_FILE="$OUT_DIR$MODEL_NAME.hypotheses.txt"
python2.7 generate_caps_text.py $DATA_DIR $OUT_DIR$MODEL_NAME $OUT_FILE
