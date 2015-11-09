DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../models/word2vec/" && pwd )"
cd $DIR

DATASET="/media/Data/flipvanrijn/datasets/text/enwiki-text"
SIZE=512
OUTPUT="/media/Data/flipvanrijn/models/word2vec/enwiki-latest-pages.${SIZE}.bin"

./word2vec -train $DATASET -output $OUTPUT -size $SIZE -binary 1