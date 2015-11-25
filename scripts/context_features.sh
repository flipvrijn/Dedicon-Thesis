DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../dataset/coco/" && pwd )"
cd $DIR

#WTV="/media/Data/flipvanrijn/models/word2vec/enwiki-latest-pages.512.bin"

IN_TRAIN="/media/Data/flipvanrijn/datasets/coco/processed/reduced/context_train_filtered_stemmed.npz"
OUT_TRAIN="/media/Data/flipvanrijn/datasets/coco/processed/full/context_train.npz"

IN_VAL="/media/Data/flipvanrijn/datasets/coco/processed/reduced/context_val_filtered_stemmed.npz"
OUT_VAL="/media/Data/flipvanrijn/datasets/coco/processed/full/context_val.npz"

echo 'Extracting context features [VAL]:'
python2.7 context_features_encode.py $IN_VAL $OUT_VAL

echo 'Condensing context features [VAL]:'
python2.7 reduce_sparse_dataset.py val /media/Data/flipvanrijn/datasets/coco/processed/full /media/Data/flipvanrijn/datasets/coco/processed/reduced --co

echo 'Extracting context features [TRAIN]:'
python2.7 context_features_encode.py $IN_TRAIN $OUT_TRAIN

echo 'Condensing context features [TRAIN]:'
python2.7 reduce_sparse_dataset.py train /media/Data/flipvanrijn/datasets/coco/processed/full /media/Data/flipvanrijn/datasets/coco/processed/reduced --co