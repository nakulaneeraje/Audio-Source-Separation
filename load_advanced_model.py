import skimage.io
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.io import imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2 

import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

def custom_loss(y_true, y_pred):    
    L1    =      tf.math.abs(tf.math.subtract(y_true, y_pred))
    sample = tf.reduce_sum(L1, [1,2])
    mean_sample = tf.math.reduce_mean(sample, 0)
    x = tf.constant([0.3048, 0.2417, 0.2092, 0.2442], tf.float32, name='x')
    print("Hey")
    return tf.tensordot(x , mean_sample, 1)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_audio_separate.h5")  
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss=custom_loss, optimizer='adam', metrics=['mae'])





# Check its architecture
# new_model.summary()

S = np.load('test_2022.npy', allow_pickle = True)
S = S[:, :, np.newaxis]
S = np.power(10, S/20)
S = S[np.newaxis, :, :, :]
pred_S = loaded_model.predict(S, verbose=1)

np.save('pred_s_2022.npy', pred_S)

