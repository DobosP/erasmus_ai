# Image Processing

## Run CNN Training:
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

##Training the CNN logs:
###Training 1:
	[INFO] loading pet features...
	[INFO] processed features
	[INFO] loading pet images...
	[INFO] processed images
	[INFO] training model...
	Train on 52479 samples, validate on 5832 samples
	Epoch 1/10
	52479/52479 [==============================] - 774s 15ms/step - loss: 1893661.1004 - val_loss: 45261.4873
	Epoch 2/10
	52479/52479 [==============================] - 687s 13ms/step - loss: 34250.5647 - val_loss: 34817.0180
	Epoch 3/10
	52479/52479 [==============================] - 1395s 27ms/step - loss: 23359.1486 - val_loss: 30943.7951
	Epoch 4/10
	52479/52479 [==============================] - 1341s 26ms/step - loss: 22074.0238 - val_loss: 43499.0763
	Epoch 5/10
	52479/52479 [==============================] - 1330s 25ms/step - loss: 21502.2411 - val_loss: 38735.2082
	Epoch 6/10
	52479/52479 [==============================] - 1317s 25ms/step - loss: 21477.0331 - val_loss: 30214.5981
	Epoch 7/10
	52479/52479 [==============================] - 1642s 31ms/step - loss: 21879.9006 - val_loss: 10931.4800
	Epoch 8/10
	52479/52479 [==============================] - 1308s 25ms/step - loss: 19653.7034 - val_loss: 20987.1016
	Epoch 9/10
	52479/52479 [==============================] - 986s 19ms/step - loss: 22931.2753 - val_loss: 6634.1201
	Epoch 10/10
	52479/52479 [==============================] - 682s 13ms/step - loss: 18439.4731 - val_loss: 21271.4816
	[INFO] predicting pet adoption speed...
	[INFO] avg. adoption speed: 2.49, std adoption speed: 1.12
	[INFO] mean: inf%, std: nan%		
	
	
