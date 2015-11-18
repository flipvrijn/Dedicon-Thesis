DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

WTV="/media/Data/flipvanrijn/models/word2vec/enwiki-latest-pages.512.bin"

IN_TRAIN="/media/Data/flipvanrijn/datasets/coco/processed/reduced/coco_train_context.pkl"
CAPS_TRAIN="/media/Data/flipvanrijn/datasets/coco/processed/reduced/captions_train2014.json"
IMG_LIST_TRAIN="/media/Data/flipvanrijn/datasets/coco/processed/reduced/train2014list.txt"
OUT_TRAIN="/media/Data/flipvanrijn/datasets/coco/processed/reduced/train_context.npz"

IN_VAL="/media/Data/flipvanrijn/datasets/coco/processed/reduced/coco_val_context.pkl"
CAPS_VAL="/media/Data/flipvanrijn/datasets/coco/processed/reduced/captions_val2014.json"
IMG_LIST_VAL="/media/Data/flipvanrijn/datasets/coco/processed/reduced/val2014list.txt"
OUT_VAL="/media/Data/flipvanrijn/datasets/coco/processed/reduced/val_context.npz"

echo 'Extracting context features [TRAIN]:'
python2.7 context_features.py --w2v=$WTV $IN_TRAIN $CAPS_TRAIN $IMG_LIST_TRAIN $OUT_TRAIN

echo 'Extracting context features [VAL]:'
python2.7 context_features.py --w2v=$WTV $IN_VAL $CAPS_VAL $IMG_LIST_VAL $OUT_VAL