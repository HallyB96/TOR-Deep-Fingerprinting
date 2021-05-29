#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Datareader import LoadDataWTFPADOW_training


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
opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer
num_classes = 96 
data_shape = (sequence_length,1)


# In[ ]:


X_train, y_train, X_valid, y_valid = LoadDataWTFPADOW_training()


# In[ ]:


X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')

X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]

print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'validation samples')


# In[ ]:


y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)


# In[ ]:


from WTFPADModel import NeuralNet
model = NeuralNet.build(input_shape=data_shape, classes=num_classes)


# In[ ]:


model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
print ("Model compiled")


# In[ ]:


history = model.fit(X_train, y_train,
		batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_data=(X_valid, y_valid), use_multiprocessing=True)


# In[ ]:


EXP_Type = 'OpenWorld_WTFPAD'
savedpath ='./saved_trained_models/%s.h5'%str(EXP_Type)
model.save(savedpath)
print("Saving Model Done!", savedpath)


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use(['bmh'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('OW-WTF-PAD model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('OW-WTF-PAD model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[ ]:




