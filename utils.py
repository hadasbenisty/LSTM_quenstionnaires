# -*- coding: utf-8 -*-
"""
Created on Sun May 29 09:56:11 2022

@author: hadas
"""
import tensorflow as tf
import keras
import numpy as np
from scipy.io import savemat

tf.config.run_functions_eagerly(False)
auc = keras.metrics.AUC(name='auc')
acc = keras.metrics.BinaryAccuracy(name='accuracy')
def get_mask(target_value):
    ### X_train ->[batch, 30, 8]
    mask_bool = target_value != -1 # [batch, 30]
    mask_array = tf.cast(mask_bool, tf.float32)
    return mask_array, mask_bool

def AUC_metric(y_true, y_pred):
  y_true = tf.reshape(y_true, [-1])
  y_pred = tf.reshape(y_pred, [-1])
  mask_array, mask_bool = get_mask(y_true)
  fil_y_true = y_true[mask_bool]
  fil_y_pred = y_pred[mask_bool]
  auc.reset_state()
  auc.update_state(fil_y_true, fil_y_pred)
  return auc.result()

def ACC_metric(y_true, y_pred):
  y_true = tf.reshape(y_true, [-1])
  y_pred = tf.reshape(y_pred, [-1])
  mask_array, mask_bool = get_mask(y_true)
  fil_y_true = y_true[mask_bool]
  fil_y_pred = y_pred[mask_bool]
  acc.reset_state()
  acc.update_state(fil_y_true, fil_y_pred)
  return acc.result()
class regression_performance():    
    def __init__(self, train_r2, test_r2, opp_r2, feature_weights, conf_all, r2_all_train, r2_all_dev):
        self.train_r2 = train_r2
        self.test_r2 = test_r2
        self.opp_r2 = opp_r2
        self.feature_weights = feature_weights
        self.conf_all = conf_all
        self.r2_all_train = r2_all_train
        self.r2_all_dev = r2_all_dev
        
class classification_performance():    
    def __init__(self, train_accuracy, chance_train, test_accuracy, chance_test, train_auc, test_auc, feature_weights, accuracy_opp, chance_level_opp, auc_opp):
        self.train_accuracy = train_accuracy
        self.chance_train = chance_train
        self.test_accuracy = test_accuracy
        self.chance_test = chance_test
        self.train_auc = train_auc
        self.test_auc = test_auc
        self.feature_weights = feature_weights
        self.accuracy_opp = accuracy_opp
        self.chance_level_opp = chance_level_opp
        self.auc_opp = auc_opp
def get_chance_level(Y):
  chance_level = np.sum(Y)/Y.size  

  #to ensure we are prediciting the majority class
  if chance_level<0.5: 
    chance_level = 1 - chance_level
  return chance_level
def get_res_file_name(filenamepath, tobalanceMinority, sampling_over, seq = 0):
    filenamepath += "_seq"
    filenamepath += str(seq)
    if tobalanceMinority:
        filenamepath += "_balance"
        if sampling_over:
            filenamepath += "_over"
        else:
            filenamepath += "under"    
    filenamepath += "10.mat"
    return filenamepath
def save_results(filenamepath, tobalanceMinority, sampling_over, seq, res):
    
    filenamepath = get_res_file_name(filenamepath, tobalanceMinority, sampling_over, seq)
    
    
    mdic={"conf_all": res.conf_all, "feature_weights": res.feature_weights, "opp_r2": res.opp_r2, 
          "train_r2": res.train_r2, "test_r2": res.test_r2, "r2_all_train": res.r2_all_train, 
          "r2_all_dev": res.r2_all_dev}
    print("saving "+filenamepath)
    
    
    
    savemat(filenamepath, mdic)    

