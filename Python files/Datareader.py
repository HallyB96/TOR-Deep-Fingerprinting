#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import _pickle as pickle
print(pickle.__doc__)


# In[17]:


def LoadDataNoDefCW():

    print ("Loading non-defended dataset for closed-world scenario")
    dataset_dir = 'dataset/NoDefCW/'


    with open(dataset_dir + 'X_train_NoDef_p3.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_NoDef_p3.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    with open(dataset_dir + 'X_valid_NoDef_p3.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_NoDef_p3.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef_p3.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_NoDef_p3.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle))


    return X_train, y_train, X_valid, y_valid, X_test, y_test

def LoadDataWalkieTalkieCW():

    dataset_dir = 'dataset/WalkieTalkieCW/'

    with open(dataset_dir + 'X_train_WalkieTalkie_p3.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_WalkieTalkie_p3.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    with open(dataset_dir + 'X_valid_WalkieTalkie_p3.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_WalkieTalkie_p3.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    with open(dataset_dir + 'X_test_WalkieTalkie_p3.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_WalkieTalkie_p3.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle))


    return X_train, y_train, X_valid, y_valid, X_test, y_test


def LoadDataWTFPADCW():

    dataset_dir = 'dataset/WtfPadCW/'

    with open(dataset_dir + 'X_train_WTFPAD_p3.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_WTFPAD_p3.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    with open(dataset_dir + 'X_valid_WTFPAD_p3.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_WTFPAD_p3.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    with open(dataset_dir + 'X_test_WTFPAD_p3.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_WTFPAD_p3.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle))


    return X_train, y_train, X_valid, y_valid, X_test, y_test



def LoadDataNoDefOW_training():

   
    dataset_dir = 'dataset/NoDefOW/'

    with open(dataset_dir + 'X_train_NoDef_p3.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_NoDef_p3.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    with open(dataset_dir + 'X_valid_NoDef_p3.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_NoDef_p3.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    return X_train, y_train, X_valid, y_valid


def LoadDataNoDefOW_Evaluation():

    dataset_dir = 'dataset/NoDefOW/'

    with open(dataset_dir + 'X_test_Mon_NoDef_p3.pkl', 'rb') as handle:
        X_test_Mon = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_Mon_NoDef_p3.pkl', 'rb') as handle:
        y_test_Mon = np.array(pickle.load(handle))

    with open(dataset_dir + 'X_test_Unmon_NoDef_p3.pkl', 'rb') as handle:
        X_test_Unmon = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_Unmon_NoDef_p3.pkl', 'rb') as handle:
        y_test_Unmon = np.array(pickle.load(handle))

    X_test_Mon = np.array(X_test_Mon)
    y_test_Mon = np.array(y_test_Mon)
    X_test_Unmon = np.array(X_test_Unmon)
    y_test_Unmon = np.array(y_test_Unmon)

  
    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon


def LoadDataWTFPADOW_training():


    dataset_dir = 'dataset/WtfPadOW/'

    with open(dataset_dir + 'X_train_WTFPAD_p3.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_WTFPAD_p3.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    with open(dataset_dir + 'X_valid_WTFPAD_p3.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_WTFPAD_p3.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    return X_train, y_train, X_valid, y_valid


def LoadDataWTFPADOW_Evaluation():

    dataset_dir = 'dataset/WtfPadOW/'


    with open(dataset_dir + 'X_test_Mon_WTFPAD_p3.pkl', 'rb') as handle:
        X_test_Mon = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_Mon_WTFPAD_p3.pkl', 'rb') as handle:
        y_test_Mon = np.array(pickle.load(handle))

    with open(dataset_dir + 'X_test_Unmon_WTFPAD_p3.pkl', 'rb') as handle:
        X_test_Unmon = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_Unmon_WTFPAD_p3.pkl', 'rb') as handle:
        y_test_Unmon = np.array(pickle.load(handle))

    X_test_Mon = np.array(X_test_Mon)
    y_test_Mon = np.array(y_test_Mon)
    X_test_Unmon = np.array(X_test_Unmon)
    y_test_Unmon = np.array(y_test_Unmon)

  
    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon

def LoadDataWalkieTalkieOW_training():

    print ("Loading Walkie-Talkie dataset for open-world scenario")
    # Point to the directory storing data
    dataset_dir = 'dataset/WalkieTalkieOW/'


    with open(dataset_dir + 'X_train_WalkieTalkie_p3.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_WalkieTalkie_p3.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    with open(dataset_dir + 'X_valid_WalkieTalkie_p3.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_WalkieTalkie_p3.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))



    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)


    return X_train, y_train, X_valid, y_valid


def LoadDataWalkieTalkieOW_Evaluation():

    dataset_dir = 'dataset/WalkieTalkieOW/'

    with open(dataset_dir + 'X_test_Mon_WalkieTalkie_p3.pkl', 'rb') as handle:
        X_test_Mon = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_Mon_WalkieTalkie_p3.pkl', 'rb') as handle:
        y_test_Mon = np.array(pickle.load(handle))

    with open(dataset_dir + 'X_test_Unmon_WalkieTalkie_p3.pkl', 'rb') as handle:
        X_test_Unmon = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_Unmon_WalkieTalkie_p3.pkl', 'rb') as handle:
        y_test_Unmon = np.array(pickle.load(handle))

    X_test_Mon = np.array(X_test_Mon)
    y_test_Mon = np.array(y_test_Mon)
    X_test_Unmon = np.array(X_test_Unmon)
    y_test_Unmon = np.array(y_test_Unmon)

  
    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon


