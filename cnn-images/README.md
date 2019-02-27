# Image Processing

## Train the model:
	- change hardcoded paths
		1. __images directory path__
		
		2. __csv document path__ 
		The cnn will use 3 columns in this document: 
			- 'PhotoAmt'
			- 'PetID'
			- 'AdoptionSpeed'
		If this is the first run, or if you do not have the resized images on your computer:
		
		3. Only if you do not already have the resized (preprocessed) images:
		__the resized images directory path__ (inside the resize_image() method)
		Make sure you created the destination folder for the resized images.
		Otherwise, if you already have the resized directory, just make sure that you call the read_image() method.
		
		
