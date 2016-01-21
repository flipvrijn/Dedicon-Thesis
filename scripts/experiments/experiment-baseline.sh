DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../models/attention/" && pwd )"
cd $DIR

DATA_DIR="/media/Data/flipvanrijn/datasets/coco/processed/full/"
OUT_DIR="/media/Data/flipvanrijn/models/"
MODEL_NAME="baseline.npz"

python2.7 eval.py --attn_type=deterministic --type=normal $DATA_DIR $OUT_DIR $MODEL_NAME

# Generate caps
OUT_FILE="$OUT_DIR$MODEL_NAME.hypotheses.txt"
python2.7 generate_caps.py $DATA_DIR $OUT_DIR$MODEL_NAME $OUT_FILE
