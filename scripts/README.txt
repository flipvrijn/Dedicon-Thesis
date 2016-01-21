This directory holds shell scripts that perform certain steps in the pipeline. The purpose of each script is explained in more detail below.

* download-coco.sh: Downloads files required to the 'downloads' directory in the ROOT. The following files are being downloaded:
	- MSCOCO instances (train, val) files from 2014
	- MSCOCO captions (train, val) files from 2014
	Example usage:
		./download.sh

* preprocess-context.sh: Preprocesses the MSCOCO files in two steps:
	- The first step is adding context data to each image (if possible) and writes the resulting dataset to a new file '*_completed.json'. 
	- Finally, the preprocessing sanitizes the newly built dataset from HTML and weird words, tokenizes the new data into separate words and writes the resulting dataset to a new file '*_sanitized.json'.
	Example usage:
		./preprocess.sh

* context-features.sh: Extracts features from the preprocessed context. NOTE: This shell script is tightly coupled with the workings of the Python script and requires tweaking the parameters for the script. 
	Example usage:
		./context-features.sh
	which will create a feature file
