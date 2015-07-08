This directory holds shell scripts that perform certain steps in the pipeline. The purpose of each script is explained in more detail below.

* download.sh: Downloads files required to the 'downloads' directory in the ROOT. The following files are being downloaded:
	- Fast RCNN models
	- MSCOCO instances (train, val) files from 2014
	- MSCOCO captions (train, val) files from 2014
	Example usage:
		./download.sh

* preprocess.sh: Preprocesses the MSCOCO files in two steps:
	- The first step is adding context data to each image (if possible) and writes the resulting dataset to a new file '*_completed.json'. 
	- Finally, the preprocessing sanitizes the newly built dataset from HTML and weird words, tokenizes the new data into separate words and writes the resulting dataset to a new file '*_sanitized.json'.
	Example usage:
		./preprocess.sh

* features.sh: Applies selective search to the MSCOCO image dataset. 
	Example usage:
		./features.sh train
	which will create a selective_search_train.mat file

* gpu.sh: Displays the GPU load and its memory usage.
	Example usage:
		./gpu.sh

* train_rcnn.sh: Will train the R-CNN using the MSCOCO train dataset. Note: this will run for a LONG time. 