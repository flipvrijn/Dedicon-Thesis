from feature_extractor import *
import caffe
import numpy as np
import pandas
import argparse

caffe.set_mode_cpu()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',dest='input_file',help='Path to image.')
    parser.add_argument('-n', dest='n', type=str, help='Top N results.')

    args = parser.parse_args()

    input_file = args.input_file
    n = args.n

    net = caffe.Classifier('models/VGG_ILSVRC_16_layers_full_deploy.prototxt' , 'models/VGG_ILSVRC_16_layers.caffemodel' , 
        image_dims = (224,224) , 
        raw_scale = 255, 
        channel_swap=(2,1,0),
        mean = np.array([103.939, 116.779, 123.68]) )

    with open('../caffe/data/ilsvrc12/synset_words.txt') as fh:
        labels = np.asmatrix([''.join(l.split(' ')[1:]).strip() for l in fh.readlines()]).transpose()

    img  = 'input/dogdinner.jpg'
    imgs = [np.array(caffe.io.load_image(img)) for x in range(0, 10)]

    net_input = np.asarray([preprocess_image(in_) for in_ in imgs])

    preds = net.forward(data = net_input)
    preds = preds[net.outputs[0]].transpose()
    s = pandas.Series(preds[:,0:1].transpose()[0], index=labels)

    print s.order(ascending=False)[:int(n)]