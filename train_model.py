from statistics import mean
from tqdm import tqdm
import random
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2
import glob
import os
from keras.models import model_from_json

TRUTH_PATH = r'C:/Users/Videh Aggarwal/Advanced_ML_Dataset_Full/Ground_Truth/'
TRAIN_PATH = r'C:/Users/Videh Aggarwal/Advanced_ML_Dataset_Full/Train/'
seed = 42
np.random.seed = seed

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1

TRUTH_HEIGHT = 256
TRUTH_WIDTH = 256

train_ids = next(os.walk(TRAIN_PATH))[2]
truth_ids = next(os.walk(TRUTH_PATH))[2]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1))
Y_train = np.zeros((len(truth_ids), TRUTH_HEIGHT , TRUTH_WIDTH, 4))

    
#import os
print('Resizing training images')
directory = TRAIN_PATH
files = sorted(glob.glob(TRAIN_PATH + '*.npy'))
n =0
for filename in files:
    S = np.load(filename, allow_pickle = True)
    S = S[:, :, np.newaxis]
    X_train[n] = np.power(10, S/20)
    n = n+1 

print('Resizing truth images')
directory = TRUTH_PATH
files = sorted(glob.glob(TRUTH_PATH + '*.npy'))
n = 0
for filename in files:
    S = np.load(filename, allow_pickle = True)
    Y_train[n] = np.power(10,S/20)
    n = n+1   
    
    
    
    
    
print(Y_train[0, :, :, 1])
print(Y_train[0, :, :, 2])
print(Y_train[0, :, :, 3])
print("Now X_train")
print(X_train[0, :, :, 0])
print('Done!')

X_train = X_train[0:1000, :, :, :]
Y_train = Y_train[0:1000, :, :, :]


inputs = keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, 1))

input_rep = keras.backend.repeat_elements(inputs, 4, axis = 3)
log_inputs = keras.backend.log(inputs)

#Contraction path
c0 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(log_inputs)
c0 = keras.layers.Dropout(0.1)(c0)
c0 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c0)
p0 = keras.layers.MaxPooling2D((2, 2))(c0)

c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p0)
c1 = keras.layers.Dropout(0.1)(c1)
c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = keras.layers.MaxPooling2D((2, 2))(c1)

c2 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = keras.layers.Dropout(0.1)(c2)
c2 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = keras.layers.Dropout(0.1)(c3)
c3 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = keras.layers.Dropout(0.1)(c4)
c4 = keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
# c5 = keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
# c5 = keras.layers.Dropout(0.1)(c5)
# c5 = keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
# p5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c5)

# c10 = keras.layers.Conv2D(2048, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
# c10 = keras.layers.Dropout(0.1)(c10)
# c10 = keras.layers.Conv2D(2048, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)

#Expansive path 
# u11 = keras.layers.Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='valid')(c10)
# #c4 = tf.image.resize(c4, (120, 120))
# u11 = keras.layers.concatenate([u11, c5])
# c11 = keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)
# c11 = keras.layers.Dropout(0.1)(c11)
# c11 = keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)

# u6 = keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='valid')(c11)
# #c4 = tf.image.resize(c4, (120, 120))
# u6 = keras.layers.concatenate([u6, c4])
# c6 = keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
# c6 = keras.layers.Dropout(0.1)(c6)
# c6 = keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='valid')(c4)
u7 = keras.layers.concatenate([u7, c3])
c7 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = keras.layers.Dropout(0.1)(c7)
c7 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid')(c3)
u8 = keras.layers.concatenate([u8, c2])
c8 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = keras.layers.Dropout(0.1)(c8)
c8 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid')(c2)
u9 = keras.layers.concatenate([u9, c1])
c9 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = keras.layers.Dropout(0.1)(c9)
c9 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

u12 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='valid')(c9)
u12 = keras.layers.concatenate([u12, c0])
c12 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u12)
c12 = keras.layers.Dropout(0.1)(c12)
c12 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c12)


    
 
temp_outputs = keras.layers.Conv2D(4, (3, 3), activation='sigmoid', kernel_initializer='he_normal', padding='same')(c12)

outputs =  keras.layers.Multiply()([input_rep, temp_outputs])

 

def custom_loss(y_true, y_pred):
        
        L1    =      tf.math.abs(tf.math.subtract(y_true, y_pred))
        sample = tf.reduce_sum(L1, [1,2])
        mean_sample = tf.math.reduce_mean(sample, 0)
        x = tf.constant([0.3048, 0.2417, 0.2092, 0.2442], tf.float32, name='x')
        print("Hey")
        return tf.tensordot(x , mean_sample, 1)
        

model = keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer= 'adam', loss= custom_loss, metrics=['mae'])
model.summary()  
results = model.fit(X_train, Y_train, epochs=4, batch_size= 16)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5

model.save('./model_audio_separation_tf', save_format='tf')
model.save_weights("model_audio_separate_filter_512.h5")
print("Saved model to disk")