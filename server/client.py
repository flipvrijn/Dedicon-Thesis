import socket
import sys
import cPickle
import numpy as np
import struct

import sys
sys.path.insert(0, '/home/flipvanrijn/Workspace/Dedicon-Thesis/networks/arctic-captions/')

import skimage
import skimage.transform
import skimage.io

from PIL import Image

import capgen
import generate_caps as gencaps
import flickr8k
import flickr30k
import coco

import matplotlib.pyplot as plt

from IPython import embed

def load_image(filename, resize=256, crop=224):
    image = Image.open(filename)
    width, height = image.size

    if width > height:
        width = (width * resize) / height
        height = resize
    else:
        height = (height * resize) / width
        width = resize
    left = (width  - crop) / 2
    top  = (height - crop) / 2
    image_resized = image.resize((width, height), Image.BICUBIC).crop((left, top, left + crop, top + crop))
    data = np.array(image_resized.convert('RGB').getdata()).reshape(crop, crop, 3)
    data = data.astype('float32') / 255
    return data

def main():
    # Load a random sample
    datasets = {
        'flickr8k': (flickr8k.load_data, flickr8k.prepare_data),
        'flickr30k': (flickr30k.load_data, flickr30k.prepare_data),
        'coco': (coco.load_data, coco.prepare_data)
    }
    dev_list    = '/home/flipvanrijn/Workspace/Dedicon-Thesis/networks/arctic-captions/splits/coco_val.txt'
    image_path  = '/media/Data/flipvanrijn/datasets/coco/images/val/'

    flist = []
    with open(dev_list, 'r') as f:
        for l in f:
            flist.append(l.strip())

    load_data, prepare_data = datasets['coco']

    _, valid, _, worddict = load_data(False, True, False)
    print 'Data loaded'

    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    while True:
        idx = np.random.randint(0, len(valid[0])) # random image
        gt = valid[0][idx][0] # groundtruth
        context = np.array(valid[1][valid[0][idx][1]].todense()).reshape([14*14, 512]) # annotations
        img = load_image(image_path+flist[valid[0][idx][1]])
        print 'GT: ' + gt

        # Send it to the server via socket
        HOST, PORT = "localhost", 9999
        data = cPickle.dumps(context)

        # Create a socket (SOCK_STREAM means a TCP socket)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # Connect to server and send data
            sock.connect((HOST, PORT))
            data = struct.pack('>I', len(data)) + data
            sock.sendall(data)

            # Receive data from the server and shut down
            received = sock.recv(1024)
            print 'Description: ' + received
        finally:
            sock.close()

        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    main()