# Image Processing

## CNN Summary:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 64, 64, 3)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 16)        448       
_________________________________________________________________
activation_1 (Activation)    (None, 64, 64, 16)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 64, 64, 16)        64        
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 16)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 32)        4640      
_________________________________________________________________
activation_2 (Activation)    (None, 32, 32, 32)        0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 32, 32, 32)        128       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 64)        18496     
_________________________________________________________________
activation_3 (Activation)    (None, 16, 16, 64)        0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 16, 16, 64)        256       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4096)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 16)                65552     
_________________________________________________________________
activation_4 (Activation)    (None, 16)                0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 16)                64        
_________________________________________________________________
dropout_1 (Dropout)          (None, 16)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 68        
_________________________________________________________________
activation_5 (Activation)    (None, 4)                 0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5         
=================================================================
Total params: 89,721
Trainable params: 89,465
Non-trainable params: 256
_________________________________________________________________
None

## Training

Train on 52479 samples, validate on 5832 samples
Epoch 1/10
52479/52479 [==============================] - 666s 13ms/step - loss: 0.1851 - mean_squared_error: 0.1851 - rmse: 0.3229 - r_square: -1.4478 - val_loss: 0.0807 - val_mean_squared_error: 0.0807 - val_rmse: 0.2440 - val_r_square: -0.0558
Epoch 2/10
52479/52479 [==============================] - 644s 12ms/step - loss: 0.0820 - mean_squared_error: 0.0820 - rmse: 0.2453 - r_square: -0.0751 - val_loss: 0.0786 - val_mean_squared_error: 0.0786 - val_rmse: 0.2412 - val_r_square: -0.0279
Epoch 3/10
52479/52479 [==============================] - 619s 12ms/step - loss: 0.0793 - mean_squared_error: 0.0793 - rmse: 0.2440 - r_square: -0.0426 - val_loss: 0.0784 - val_mean_squared_error: 0.0784 - val_rmse: 0.2429 - val_r_square: -0.0257
Epoch 4/10
52479/52479 [==============================] - 632s 12ms/step - loss: 0.0793 - mean_squared_error: 0.0793 - rmse: 0.2444 - r_square: -0.0412 - val_loss: 0.0807 - val_mean_squared_error: 0.0807 - val_rmse: 0.2459 - val_r_square: -0.0541
Epoch 5/10
52479/52479 [==============================] - 686s 13ms/step - loss: 0.0796 - mean_squared_error: 0.0796 - rmse: 0.2455 - r_square: -0.0439 - val_loss: 0.0789 - val_mean_squared_error: 0.0789 - val_rmse: 0.2450 - val_r_square: -0.0304
Epoch 6/10
52479/52479 [==============================] - 689s 13ms/step - loss: 0.0791 - mean_squared_error: 0.0791 - rmse: 0.2449 - r_square: -0.0363 - val_loss: 0.0787 - val_mean_squared_error: 0.0787 - val_rmse: 0.2444 - val_r_square: -0.0284
Epoch 7/10
52479/52479 [==============================] - 677s 13ms/step - loss: 0.0790 - mean_squared_error: 0.0790 - rmse: 0.2448 - r_square: -0.0372 - val_loss: 0.0790 - val_mean_squared_error: 0.0790 - val_rmse: 0.2448 - val_r_square: -0.0324
Epoch 8/10
52479/52479 [==============================] - 6676s 127ms/step - loss: 0.0790 - mean_squared_error: 0.0790 - rmse: 0.2452 - r_square: -0.0367 - val_loss: 0.0786 - val_mean_squared_error: 0.0786 - val_rmse: 0.2444 - val_r_square: -0.0267
Epoch 9/10
52479/52479 [==============================] - 1167s 22ms/step - loss: 0.0787 - mean_squared_error: 0.0787 - rmse: 0.2446 - r_square: -0.0320 - val_loss: 0.0791 - val_mean_squared_error: 0.0791 - val_rmse: 0.2448 - val_r_square: -0.0327
Epoch 10/10
52448/52479 [============================>.] - ETA: 3s - loss: 0.0784 - mean_squared_error: 0.0784 - rmse: 0.2438 - r_square: -0.0289