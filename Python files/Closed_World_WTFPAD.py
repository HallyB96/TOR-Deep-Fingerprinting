#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Datareader import LoadDataWTFPADCW


# In[ ]:


import tensorflow as tf
from tensorflow.keras import backend as K
import random
from tensorflow.keras.optimizers import Adamax
import numpy as np
import os


# In[ ]:


epochs = 40  
batch_size = 128 
sequence_length = 5000 
opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 
num_classes = 95 
data_shape = (sequence_length,1)


# In[ ]:


X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataWTFPADCW()


# In[ ]:


X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('float32')

X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]
X_test = X_test[:, :,np.newaxis]

print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')


# In[ ]:


y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# In[ ]:


from WTFPADModel import NeuralNet


# In[ ]:


tf.debugging.set_log_device_placement(True)
model = NeuralNet.build(input_shape=data_shape, classes=num_classes)


# In[ ]:


model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
print ("Model compiled")


# In[ ]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[ ]:


history = model.fit(X_train, y_train,
		batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid), use_multiprocessing=True)

score_test = model.evaluate(X_test, y_test, verbose=2)
print("Testing accuracy:", score_test[1])


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use(['bmh'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('CW-WTF_PAD model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CW-WTF_PAD model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[ ]:




