import h5py
import cPickle
from progress.bar import Bar

from IPython import embed

def main():
	f_in = h5py.File('/media/Data/flipvanrijn/datasets/coco/sents_train.h5', 'r')
	f_images = '/media/Data/flipvanrijn/datasets/coco/splits/coco_train.txt'
	f_out = 'context.pkl'

	with open(f_images, 'r') as f:
		images = f.read().split()

	im2idx = dict(zip(images, range(len(images))))
	idx2im = dict(zip(im2idx.values(), im2idx.keys()))

	print 'Building old indexes...'
	old_im2idx = {}
	for i, im_name in enumerate(f_in['sentences/image_names']):
		old_im2idx[im_name] = i

	titles = []
	descriptions = []
	tags = []
	bar = Bar('Fetching', max=len(images))
	for image in images:
		old_idx = old_im2idx[image]
		titles.append(f_in['context/titles'][old_idx])
		descriptions.append(f_in['context/descriptions'][old_idx])
		tag_ids = f_in['context/tags'][old_idx].tolist()
		tag_ids = sorted(tag_ids)
		new_tags = []
		for i in tag_ids:
			new_tags.append(f_in['sentences/tags'][i])
		tags.append(new_tags)
		bar.next()
	bar.finish()

	with open(f_out, 'w') as f:
		cPickle.dump(titles, f)
		cPickle.dump(descriptions, f)
		cPickle.dump(tags, f)

if __name__ == '__main__':
	main()