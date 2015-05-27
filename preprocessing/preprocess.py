import os, fnmatch, Image, ImageOps

def findFiles(directory, pattern):
	for root, dirs, files in os.walk(directory):
		for basename in files:
			if fnmatch.fnmatch(basename, pattern):
				filename = os.path.join(root, basename)
				yield filename

def resize(filename):
	outFile = os.path.splitext(filename)[0]+'.resized.jpg'
	size = (256, 256)
	img = Image.open(filename)
	img = ImageOps.fit(img, size, Image.ANTIALIAS)
	img.save(outFile, 'JPEG')

for filename in findFiles('../data/images', '*.jpg'):
	resize(filename)