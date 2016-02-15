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

echo 'Preprocessing data set:'
python2.7 preprocess.py 

mv '/media/Data/flipvanrijn/datasets/coco/processed/reduced/coco_align.dev.pkl' '/media/Data/flipvanrijn/datasets/coco/processed/reduced/w2v2ngramplus/coco_align.dev.pkl'
mv '/media/Data/flipvanrijn/datasets/coco/processed/reduced/coco_align.train.pkl' '/media/Data/flipvanrijn/datasets/coco/processed/reduced/w2v2ngramplus/coco_align.train.pkl'
mv '/media/Data/flipvanrijn/datasets/coco/processed/reduced/coco_align.test.pkl' '/media/Data/flipvanrijn/datasets/coco/processed/reduced/w2v2ngramplus/coco_align.test.pkl'
mv '/media/Data/flipvanrijn/datasets/coco/processed/reduced/dictionary.pkl' '/media/Data/flipvanrijn/datasets/coco/processed/reduced/w2v2ngramplus/dictionary.pkl'

echo 'Start training'
python2.7 ../../models/attention/train.py --attn_type=deterministic --type=t_attn --preproc_type=w2v --preproc_params=with_stemming=1,n=2,merge=sum /media/Data/flipvanrijn/datasets/coco/processed/reduced/w2v2ngramplus /media/Data/flipvanrijn/models context_att_w2v2ngramplus.npz