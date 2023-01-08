# -*- coding: utf-8 -*-
"""
Created on Sun May 29 16:15:08 2022

@author: hadas
"""

#import keras
from os.path import exists
from os import mkdir
import scipy.io
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras import regularizers
import data_set_utils_v2 as dtut
from tensorflow.keras.optimizers import Adam
import numpy as np

import utils as ut
from sklearn.metrics import r2_score

def run_LSTM_comparison_many2one(inputkeyList, X_train, Y_train, X_test, Y_test,  
                        opposite_X_test, opposite_Y_test, epochs, 
                        batch_size, sequence_length, hidden_size, dropout, learning_rate, l1_regularizer, 
                        l2_regularizer, lstm_layers, dense_layers):
  model = Sequential() #creates a sequential model which is a plain stack of of densely connected NN layers
  model.add(LSTM(units = hidden_size, input_shape=(sequence_length, len(inputkeyList)), dropout = dropout, recurrent_dropout = dropout, kernel_regularizer=regularizers.l1(l1=l1_regularizer), stateful = False))
  #model.add(LSTM(units = hidden_size, input_shape=(sequence_length, len(inputkeyList))))

  # for n in range(lstm_layers):
  #   model.add(LSTM(hidden_size, stateful = False)) 

  for n in range(dense_layers):
    model.add(Dense(hidden_size))
  model.add(Dense(1))
   
  optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
  
  model.compile(loss='mse', optimizer=optimizer, metrics=['mse']) 
  
  model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = epochs, batch_size = batch_size, verbose = 0)
  
  #make prediction on the test dataset
  predictions_test = model.predict(X_test)
  predictions_train = model.predict(X_train)
  predictions_opposite= model.predict(opposite_X_test)
  return model, r2_score(Y_train, predictions_train), r2_score(Y_test, predictions_test), r2_score(opposite_Y_test, predictions_opposite)
  
 
def get_best_LSTM_params(w_data, epochs, batch_size, sequence_length, hidden_size, 
                                       dropout, learning_rate, l1_regularizer, l2_regularizer, lstm_layers, dense_layers):
    
    
    conf_all = []
    r2_all_train = []
    r2_all_dev = []
    for e in epochs:
        for b in batch_size:
            for s in sequence_length:
                for h in hidden_size:
                    for d in dropout:
                        for lr in learning_rate:
                            for l1 in l1_regularizer:
                                for l2 in l2_regularizer:
                                    for lstmi in lstm_layers:
                                        for densei in dense_layers:
                                            train_r2=[]
                                            dev_r2 = [] 
                                            for k in range(w_data.n_splits):
                                                X_train, Y_train = w_data.load_XY_data(w_data.trainingList[k], s)
                                                X_dev, Y_dev = w_data.load_XY_data(w_data.devList[k], s) 
                                                model, metrics1 = run_LSTM_comparison_many2one(w_data.inputkeyList, X_train, Y_train, X_dev, Y_dev, 
                                                                                    X_dev, Y_dev, 
                                                                                   e, b, s, h, 
                                                                                   d, lr, l1, l2, lstmi, densei)
                                                train_r2.append(metrics1[0])
                                                dev_r2.append(metrics1[1])
                                            r2_all_train.append(np.mean(train_r2))
                                            r2_all_dev.append(np.mean(dev_r2))
                                            conf = {"e": e, "b": b, "s": s, "h": h,
                                                    "d": d, "lr": lr, "l1": l1, "l2": l2,
                                                    "lstmi": lstmi, "densei": densei}
                                            conf_all.append(conf)
           
    return conf_all, r2_all_train, r2_all_dev
def gradient_importance(seq, model):

    seq = tf.Variable(seq[np.newaxis,:,:], dtype=tf.float32) #seq.shape = (1, 13, 8)

    with tf.GradientTape() as tape: #capture the gradients on the input
        predictions = model(seq) #predictions.shape = (1, 1)

    grads = tape.gradient(predictions, seq) #produces gradients of the same shape of the single input sequence: (13x8)
    grads = tf.reduce_mean(grads, axis=1).numpy()[0] #obtain the impact of each sequence feature as average over the time dimension
    
    return grads

def CV_LSTM(w_data, w_data_opp, sequence_length, epoch, batchsize, hidden_size, drop_out, lr, l1_reg, l2_reg, lstm_l, dense_l):
    train_r2=[]
    test_r2 = []    
    opp_r2=[]
    
    #act_array = []
    grad_array = []
    
    #going through each fold, train and test    
    for k in range(w_data.n_splits):    
        X_train0, Y_train0 = w_data.load_XY_data(w_data.trainingList0[k], sequence_length)
        X_test, Y_test = w_data.load_XY_data(w_data.testingList[k], sequence_length)     
        #getting data from our other population of interest
        opposite_X_test, opposite_Y_test = w_data_opp.load_XY_data(w_data_opp.testingList[k], sequence_length)
      
        metrics1 = run_LSTM_comparison_many2one(w_data.inputkeyList, X_train0, Y_train0, X_test, Y_test, 
                                        opposite_X_test, opposite_Y_test, 
                                       epoch, batchsize, sequence_length, hidden_size, 
                                       drop_out, lr, l1_reg, l2_reg, 
                                       lstm_l, dense_l)
   
        
        best_model = metrics1[0]
        train_r2.append(metrics1[1])
        test_r2.append(metrics1[2])
        opp_r2.append(metrics1[3])
        
        grad_array_temp = np.asarray([0] * len(X_test[0][0]))[np.newaxis, :]
        examples = len(X_test)
        for e in range(examples):
          grad_array_temp = grad_array_temp + gradient_importance(X_test[e], best_model)[np.newaxis, :]
          #act_array_temp = act_array_temp + activation_grad(X_test[e], best_model)[np.newaxis, :]
        grad_array.append(grad_array_temp / examples)
    
    
       
    res = ut.regression_performance(train_r2, test_r2, opp_r2, grad_array, conf_all = 0, r2_all_train = 0, r2_all_dev = 0) 
    print("train: " + str(np.mean(train_r2)) + " test: " + str(np.mean(test_r2)))
    return res



def do_analysis(use_cdisum, do_deviations, do_normalize, n_splits, tobalanceMinority, sampling_over, datain, outdata, outdir):

    waves_files = ["../data/ESMdata_w1_v9.csv", "../data/ESMdata_w2_v11.0.csv", "../data/ESMdata_w3_v2_all.csv"]
    female_files = ["../data/ESMdata_w1_v9_female.csv", "../data/ESMdata_w2_v11.0_female.csv", "../data/ESMdata_w3_v2_female_all.csv"]
    male_files = ["../data/ESMdata_w1_v9_male.csv", "../data/ESMdata_w2_v11.0_male.csv", "../data/ESMdata_w3_v2_male_all.csv"]
    young_files = ["../data/ESMdata_w1_v9_young.csv", "../data/ESMdata_w2_v11.0_young.csv", "../data/ESMdata_w3_v2_young_all.csv"]
    old_files = ["../data/ESMdata_w1_v9_old.csv", "../data/ESMdata_w2_v11.0_old.csv", "../data/ESMdata_w3_v2_old_all.csv"]
    
    w_inputkeyList = [];
    w_inputkeyList.append(["pos_inter_mom_sum", "pos_inter_dad_sum","pos_inter_sibling_sum",'pos_inter_friend_part_sum', "neg_inter_mom_sum","neg_inter_dad_sum","neg_inter_sibling_sum","neg_inter_friend_part_sum"])
    w_inputkeyList.append(["pos_inter_mom_sum", "pos_inter_dad_sum","pos_inter_sibling_sum","pos_inter_friend_sum", "neg_inter_mom_sum","neg_inter_dad_sum","neg_inter_sibling_sum","neg_inter_friend_sum"])
    w_inputkeyList.append(["pos_inter_mom_sum", "pos_inter_dad_sum","pos_inter_sibling_sum","pos_inter_friend_sum", "neg_inter_mom_sum","neg_inter_dad_sum","neg_inter_sibling_sum","neg_inter_friend_sum"])
    
    
    w_data=[]    
    do_again = False
    for file, keylist in zip(waves_files, w_inputkeyList):        
        w_data.append(dtut.w_data(use_cdisum, do_deviations, do_normalize, file, keylist, "CDI_sum", tobalanceMinority, sampling_over, n_splits))
    filenamepath = outdir + "/lstm_w" + str(datain+1) + "_w" + str(outdata+1)
    print(filenamepath)
    path_to_file = ut.get_res_file_name(filenamepath, tobalanceMinority, sampling_over)
    file_exists = exists(path_to_file)
    if use_cdisum:
        gs = scipy.io.loadmat('grid_search_summary_waves.mat')
    else:
        gs = scipy.io.loadmat('grid_search_summary_waves_noCDI.mat')
        
    if not file_exists or do_again:
        v1 = gs["datastrin"] == "w" + str(datain+1)  
        ind = np.where(v1)
        res = CV_LSTM(w_data[datain], w_data[outdata], gs["s"][ind][0], gs["ep"][ind][0], gs["b"][ind][0], 
                          gs["h"][ind][0], gs["d"][ind][0], gs["lr"][ind][0], 
                          gs["l1"][ind][0], gs["l2"][ind][0], 
                          gs["lstmi"][ind][0],gs["densei"][ind][0])
        ut.save_results(filenamepath, tobalanceMinority, sampling_over, 0, res)
        
        
    w_female_data=[]    
    for file, keylist in zip(female_files, w_inputkeyList):  
        w_female_data.append(dtut.w_data(use_cdisum, do_deviations, do_normalize, file, keylist, "CDI_sum", tobalanceMinority, sampling_over, n_splits))
    w_male_data=[]
    for file, keylist in zip(male_files, w_inputkeyList):  
        w_male_data.append(dtut.w_data(use_cdisum, do_deviations, do_normalize, file, keylist, "CDI_sum", tobalanceMinority, sampling_over, n_splits))
    
    w_young_data=[]
    for file, keylist in zip(young_files, w_inputkeyList):  
        w_young_data.append(dtut.w_data(use_cdisum, do_deviations, do_normalize, file, keylist, "CDI_sum", tobalanceMinority, sampling_over, n_splits))
    w_old_data=[]
    for file, keylist in zip(old_files, w_inputkeyList):  
        w_old_data.append(dtut.w_data(use_cdisum, do_deviations, do_normalize, file, keylist, "CDI_sum", tobalanceMinority, sampling_over, n_splits))
    
  
    
     
    # gender 
    if use_cdisum:
        gs = scipy.io.loadmat('grid_search_summary_fe.mat')
    else:
        gs = scipy.io.loadmat('grid_search_summary_fe_noCDI.mat')
        
    filenamepath = outdir + "/lstm_fe" + str(datain+1) + "_m" + str(outdata+1)               
    print(filenamepath)
    path_to_file = ut.get_res_file_name(filenamepath, tobalanceMinority, sampling_over, 0)
    file_exists = exists(path_to_file)
    if not file_exists or do_again:
        v1 = gs["datastrin"] == "fe" + str(datain+1)  
        ind = np.where(v1)
        res = CV_LSTM(w_female_data[datain], w_male_data[outdata], gs["s"][ind][0], gs["ep"][ind][0], gs["b"][ind][0], 
                          gs["h"][ind][0], gs["d"][ind][0], gs["lr"][ind][0], 
                          gs["l1"][ind][0], gs["l2"][ind][0], 
                          gs["lstmi"][ind][0],gs["densei"][ind][0])
        
        ut.save_results(filenamepath, tobalanceMinority, sampling_over, 0, res)
    
    if use_cdisum:
        gs = scipy.io.loadmat('grid_search_summary_m.mat')
    else:
        gs = scipy.io.loadmat('grid_search_summary_m_noCDI.mat')
    filenamepath = outdir + "/lstm_m" + str(datain+1) + "_fe" + str(outdata+1)
    print(filenamepath)
    path_to_file = ut.get_res_file_name(filenamepath, tobalanceMinority, sampling_over, 0)
    file_exists = exists(path_to_file)
    if not file_exists or do_again:
        v1 = gs["datastrin"] == "m" + str(datain+1)  
        ind = np.where(v1)
        res = CV_LSTM(w_male_data[datain], w_female_data[outdata], gs["s"][ind][0], gs["ep"][ind][0], gs["b"][ind][0], 
                          gs["h"][ind][0], gs["d"][ind][0], gs["lr"][ind][0], 
                          gs["l1"][ind][0], gs["l2"][ind][0], 
                          gs["lstmi"][ind][0],gs["densei"][ind][0])
        
        ut.save_results(filenamepath, tobalanceMinority, sampling_over, 0, res)
    
    
    
    # age 
    if use_cdisum:
        gs = scipy.io.loadmat('grid_search_summary_young.mat')
    else:
        gs = scipy.io.loadmat('grid_search_summary_young_noCDI.mat')
        
    filenamepath = outdir + "/lstm_young" + str(datain+1) + "_old" + str(outdata+1)
    print(filenamepath)
    path_to_file = ut.get_res_file_name(filenamepath, tobalanceMinority, sampling_over, 0)
    file_exists = exists(path_to_file)
    if not file_exists or do_again:
        v1 = gs["datastrin"] == "young" + str(datain+1)  
        ind = np.where(v1)
        res = CV_LSTM(w_young_data[datain], w_old_data[outdata], gs["s"][ind][0], gs["ep"][ind][0], gs["b"][ind][0], 
                          gs["h"][ind][0], gs["d"][ind][0], gs["lr"][ind][0], 
                          gs["l1"][ind][0], gs["l2"][ind][0], 
                          gs["lstmi"][ind][0],gs["densei"][ind][0])
        
        ut.save_results(filenamepath, tobalanceMinority, sampling_over, 0, res)
    filenamepath = outdir + "/lstm_old" + str(datain+1) + "_young" + str(outdata+1)
    if use_cdisum:
        gs = scipy.io.loadmat('grid_search_summary_old.mat')
    else:
        gs = scipy.io.loadmat('grid_search_summary_old_noCDI.mat')
       
    print(filenamepath)
    path_to_file = ut.get_res_file_name(filenamepath, tobalanceMinority, sampling_over, 0)
    file_exists = exists(path_to_file)
    if not file_exists or do_again:
        v1 = gs["datastrin"] == "old" + str(datain+1)  
        ind = np.where(v1)
        res = CV_LSTM(w_old_data[datain], w_young_data[outdata], gs["s"][ind][0], gs["ep"][ind][0], gs["b"][ind][0], 
                          gs["h"][ind][0], gs["d"][ind][0], gs["lr"][ind][0], 
                          gs["l1"][ind][0], gs["l2"][ind][0], 
                          gs["lstmi"][ind][0],gs["densei"][ind][0])
        
        ut.save_results(filenamepath, tobalanceMinority, sampling_over, 0, res)
    
        
        
do_deviations = True
do_normalize = True
use_cdisum = True
if do_deviations:
    devstr = "with_dev"
else:
    devstr = "no_dev"
if do_normalize:
   normstr = "_norm"
else:
    normstr = ""
if use_cdisum:
    cdistr = "_with_cdioutput"
else:
    cdistr = "_no_cdioutput"
     
outdir = "../results_w3_many2one_" + devstr + normstr + cdistr + "/"
if not exists(outdir):
    mkdir(outdir)
n_splits = 10
#for indata in range(3):
for indata in range(3):
    for outdata in range(3):
        do_analysis(use_cdisum, do_deviations, do_normalize, n_splits, False, False, indata, outdata, outdir)
            #do_analysis(n_splits, True, False, indata, outdata)
            #do_analysis(n_splits, True, True, indata, outdata)  
        
        
