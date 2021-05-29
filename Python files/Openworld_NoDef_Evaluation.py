#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.models import load_model
import numpy as np


# In[ ]:


def Prediction(trained_model = None, dataset = None):
    X_test_Mon = dataset['X_test_Mon'].astype('float32')
    X_test_Unmon = dataset['X_test_Unmon'].astype('float32')
    X_test_Mon = X_test_Mon[:, :, np.newaxis]
    X_test_Unmon = X_test_Unmon[:, :, np.newaxis]
    result_Mon = trained_model.predict(X_test_Mon, verbose=2)
    result_Unmon = trained_model.predict(X_test_Unmon, verbose=2)
    return result_Mon, result_Unmon


# In[ ]:


def Evaluation(threshold_val = None, monitored_label = None,
                   unmonitored_label = None, result_Mon = None,
                   result_Unmon = None, log_file = None):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(result_Mon)):
        sm_vector = result_Mon[i]
        predicted_class = np.argmax(sm_vector)
        max_prob = max(sm_vector)

        if predicted_class in monitored_label:
            if max_prob >= threshold_val: 
                TP = TP + 1
            else:
                FN = FN + 1
        elif predicted_class in unmonitored_label: 
            FN = FN + 1

    for i in range(len(result_Unmon)):
        sm_vector = result_Unmon[i]
        predicted_class = np.argmax(sm_vector)
        max_prob = max(sm_vector)

        if predicted_class in monitored_label: 
            if max_prob >= threshold_val: 
                FP = FP + 1
            else: 
                TN = TN + 1
        elif predicted_class in unmonitored_label:
            TN = TN + 1

   
    TPR = float(TP) / (TP + FN)
    FPR = float(FP) / (FP + TN)
    Precision = float(TP) / (TP + FP)
    Recall = float(TP) / (TP + FN)
    log_file.writelines("%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n"%(threshold_val, TP, FP, TN, FN, TPR, FPR, Precision, Recall))


# In[ ]:


def OW_Evaluation():
    evaluation_type = 'OpenWorld_NoDef'
    threshold = 1.0 - 1 / np.logspace(0.05, 2, num=15, endpoint=True)
    file_name = './results/%s.csv'%evaluation_type
    log_file =  open(file_name, "w")
    dataset = {}
    model_name = ''
    from Datareader import LoadDataNoDefOW_Evaluation
    X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon = LoadDataNoDefOW_Evaluation()
    model_name = './saved_trained_models/OpenWorld_NoDef.h5'

    dataset['X_test_Mon'] = X_test_Mon
    dataset['y_test_Mon'] = y_test_Mon
    dataset['X_test_Unmon'] = X_test_Unmon
    dataset['y_test_Unmon'] = y_test_Unmon

    trained_model = load_model(model_name)
    result_Mon, result_Unmon = Prediction(trained_model = trained_model, dataset = dataset)
    monitored_label = list(y_test_Mon)
    unmonitored_label = list(y_test_Unmon)
    
    
    log_file.writelines("%s,%s,%s,%s,%s,%s  ,%s  ,  %s, %s\n" % ('Threshold', 'TP', 'FP', 'TN', 'FN', 'TPR', 'FPR', 'Precision', 'Recall'))
    for th in threshold:
        Evaluation(threshold_val = th, monitored_label = monitored_label,
                   unmonitored_label = unmonitored_label, result_Mon = result_Mon,
                   result_Unmon = result_Unmon, log_file = log_file)
    log_file.close()


# In[ ]:


OW_Evaluation()


# In[ ]:




