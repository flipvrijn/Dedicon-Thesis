DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../dataset/coco/" && pwd )"
cd $DIR

#WTV="/media/Data/flipvanrijn/models/word2vec/enwiki-latest-pages.512.bin"

IN_TRAIN="/media/Data/flipvanrijn/datasets/coco/processed/reduced/coco_train_context.pkl"
OUT_TRAIN="/media/Data/flipvanrijn/datasets/coco/processed/reduced/context_train.npz"

IN_VAL="/media/Data/flipvanrijn/datasets/coco/processed/reduced/coco_val_context.pkl"
OUT_VAL="/media/Data/flipvanrijn/datasets/coco/processed/reduced/context_val.npz"

echo 'Extracting context features [VAL]:'
python2.7 context_features.py --type=w2v -w 2 $IN_VAL $OUT_VAL

echo 'Extracting context features [TRAIN]:'
python2.7 context_features.py --type=w2v -w 2 $IN_TRAIN $OUT_TRAIN