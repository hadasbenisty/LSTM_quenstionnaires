
"""
Created on Sun May 29 16:15:08 2022

@author: hadas
"""

#import keras
from os import mkdir
from os.path import exists
import sys
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras import regularizers
import data_set_utils_v2 as dtut
from tensorflow.keras.optimizers import Adam

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
  return r2_score(Y_train, predictions_train), r2_score(Y_test, predictions_test), r2_score(opposite_Y_test, predictions_opposite)
  
 
def get_best_LSTM_params(w_data, e, b, s, h, 
                                       d, lr, l1, l2, lstmi, densei):
    
    train_r2=[]
    dev_r2 = [] 
    for k in range(w_data.n_splits):
        X_train, Y_train = w_data.load_XY_data(w_data.trainingList[k], s)
        X_dev, Y_dev = w_data.load_XY_data(w_data.devList[k], s) 
        metrics1 = run_LSTM_comparison_many2one(w_data.inputkeyList, X_train, Y_train, X_dev, Y_dev, 
                                            X_dev, Y_dev, 
                                           e, b, s, h, 
                                           d, lr, l1, l2, lstmi, densei)
        train_r2.append(metrics1[0])
        dev_r2.append(metrics1[1])
    
    conf = {"e": e, "b": b, "s": s, "h": h,
            "d": d, "lr": lr, "l1": l1, "l2": l2,
            "lstmi": lstmi, "densei": densei}
    
           
    return conf, train_r2, dev_r2

def CV_LSTM(w_data, sequence_length, epochs, batch_size, hidden_size, dropout, learning_rate, l1_regularizer, 
                l2_regularizer, lstm_layers, dense_layers):
  
    
    print("grid search")   
    
    conf_all, r2_all_train, r2_all_dev = get_best_LSTM_params(w_data, epochs, batch_size, sequence_length, hidden_size, 
                                       dropout, learning_rate, l1_regularizer, l2_regularizer, lstm_layers, dense_layers)
    
    res = ut.regression_performance(0, 0, 0, 0, conf_all, r2_all_train, r2_all_dev) 
    
    return res



def do_analysis(use_cdisum, do_deviations, do_normalize, n_splits, tobalanceMinority, sampling_over, datain, epochs, 
                batch_size, dropout, learning_rate, l1_regularizer, l2_regularizer, 
                lstm_layers, dense_layers, sequenceList, hidden_size):

    waves_files = ["../data/ESMdata_w1_v9.csv", "../data/ESMdata_w2_v11.0.csv", "../data/ESMdata_w3_v2_all.csv"]
    female_files = ["../data/ESMdata_w1_v9_female.csv", "../data/ESMdata_w2_v11.0_female.csv", "../data/ESMdata_w3_v2_female_all.csv"]
    male_files = ["../data/ESMdata_w1_v9_male.csv", "../data/ESMdata_w2_v11.0_male.csv", "../data/ESMdata_w3_v2_male_all.csv"]
    young_files = ["../data/ESMdata_w1_v9_young.csv", "../data/ESMdata_w2_v11.0_young.csv", "../data/ESMdata_w3_v2_young_all.csv"]
    old_files = ["../data/ESMdata_w1_v9_old.csv", "../data/ESMdata_w2_v11.0_old.csv", "../data/ESMdata_w3_v2_old_all.csv"]
    
    w_inputkeyList = [];
    w_inputkeyList.append(["pos_inter_mom_sum", "pos_inter_dad_sum","pos_inter_sibling_sum",'pos_inter_friend_part_sum', "neg_inter_mom_sum","neg_inter_dad_sum","neg_inter_sibling_sum","neg_inter_friend_part_sum"])
    w_inputkeyList.append(["pos_inter_mom_sum", "pos_inter_dad_sum","pos_inter_sibling_sum","pos_inter_friend_sum", "neg_inter_mom_sum","neg_inter_dad_sum","neg_inter_sibling_sum","neg_inter_friend_sum"])
    w_inputkeyList.append(["pos_inter_mom_sum", "pos_inter_dad_sum","pos_inter_sibling_sum","pos_inter_friend_sum", "neg_inter_mom_sum","neg_inter_dad_sum","neg_inter_sibling_sum","neg_inter_friend_sum"])
    
    
    if use_cdisum:
        outpath = "../grid_search_withcdi"
    else:
        outpath = "../grid_search_nocdi"
    if not exists(outpath):
        mkdir(outpath)
    w_data=[]    
    do_again = False
    for file, keylist in zip(waves_files, w_inputkeyList):        
        w_data.append(dtut.w_data(use_cdisum, do_deviations, do_normalize, file, keylist, "CDI_sum", tobalanceMinority, sampling_over, n_splits))
    filenamepath = outpath + "/lstm_w" + str(datain+1) +  "_epochs" + str(epochs)+ "_batch_size" + str(batch_size)+ "_dropout" + str(dropout)                        + "_lr" + str(learning_rate)+ "_l1" + str(l1_regularizer)+ "_l2" + str(l2_regularizer)                        + "_lstm_layers" + str(lstm_layers)+ "_dense_layers" + str(dense_layers)+                        "_sequenceList" + str(sequenceList) + "_hidden_size" + str(hidden_size)    
    print(filenamepath)
    path_to_file = ut.get_res_file_name(filenamepath, tobalanceMinority, sampling_over)
    file_exists = exists(path_to_file)
    if not file_exists or do_again:
        res = CV_LSTM(w_data[datain], sequenceList, epochs, batch_size, 
                          hidden_size, dropout, learning_rate, 
                          l1_regularizer, l2_regularizer, 
                          lstm_layers,dense_layers)
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
    filenamepath = outpath + "/lstm_fe" + str(datain+1) + "_epochs" + str(epochs)+ "_batch_size" + str(batch_size)+ "_dropout" + str(dropout)                        + "_lr" + str(learning_rate)+ "_l1" + str(l1_regularizer)+ "_l2" + str(l2_regularizer)                        + "_lstm_layers" + str(lstm_layers)+ "_dense_layers" + str(dense_layers)+                        "_sequenceList" + str(sequenceList) + "_hidden_size" + str(hidden_size)    
    
    print(filenamepath)
    path_to_file = ut.get_res_file_name(filenamepath, tobalanceMinority, sampling_over, 0)
    file_exists = exists(path_to_file)
    if not file_exists or do_again:
        res = CV_LSTM(w_female_data[datain], sequenceList, epochs, batch_size, 
                          hidden_size, dropout, learning_rate, 
                          l1_regularizer, l2_regularizer, 
                          lstm_layers,dense_layers)
        ut.save_results(filenamepath, tobalanceMinority, sampling_over, 0, res)
    filenamepath = outpath + "/lstm_m" + str(datain+1) +  "_epochs" + str(epochs)+ "_batch_size" + str(batch_size)+ "_dropout" + str(dropout)                        + "_lr" + str(learning_rate)+ "_l1" + str(l1_regularizer)+ "_l2" + str(l2_regularizer)                        + "_lstm_layers" + str(lstm_layers)+ "_dense_layers" + str(dense_layers)+                        "_sequenceList" + str(sequenceList) + "_hidden_size" + str(hidden_size)    
    
    print(filenamepath)
    path_to_file = ut.get_res_file_name(filenamepath, tobalanceMinority, sampling_over, 0)
    file_exists = exists(path_to_file)
    if not file_exists or do_again:
        res = CV_LSTM(w_male_data[datain],sequenceList, epochs, batch_size, 
                          hidden_size, dropout, learning_rate, 
                          l1_regularizer, l2_regularizer, 
                          lstm_layers,dense_layers)
        ut.save_results(filenamepath, tobalanceMinority, sampling_over, 0, res)
    
    
    
    # age
    filenamepath = outpath + "/lstm_young" + str(datain+1) +  "_epochs" + str(epochs)+ "_batch_size" + str(batch_size)+ "_dropout" + str(dropout)                        + "_lr" + str(learning_rate)+ "_l1" + str(l1_regularizer)+ "_l2" + str(l2_regularizer)                        + "_lstm_layers" + str(lstm_layers)+ "_dense_layers" + str(dense_layers)+                        "_sequenceList" + str(sequenceList) + "_hidden_size" + str(hidden_size)    
    
    print(filenamepath)
    path_to_file = ut.get_res_file_name(filenamepath, tobalanceMinority, sampling_over, 0)
    file_exists = exists(path_to_file)
    if not file_exists or do_again:
        res = CV_LSTM(w_young_data[datain], sequenceList, epochs, batch_size, 
                          hidden_size, dropout, learning_rate, 
                          l1_regularizer, l2_regularizer, 
                          lstm_layers,dense_layers)
        ut.save_results(filenamepath, tobalanceMinority, sampling_over, 0, res)
    filenamepath = outpath + "/lstm_old" + str(datain+1) + "_epochs" + str(epochs)+ "_batch_size" + str(batch_size)+ "_dropout" + str(dropout)                        + "_lr" + str(learning_rate)+ "_l1" + str(l1_regularizer)+ "_l2" + str(l2_regularizer)                        + "_lstm_layers" + str(lstm_layers)+ "_dense_layers" + str(dense_layers)+                        "_sequenceList" + str(sequenceList) + "_hidden_size" + str(hidden_size)    

    print(filenamepath)
    path_to_file = ut.get_res_file_name(filenamepath, tobalanceMinority, sampling_over, 0)
    file_exists = exists(path_to_file)
    if not file_exists or do_again:
        res = CV_LSTM(w_old_data[datain], sequenceList, epochs, batch_size, 
                          hidden_size, dropout, learning_rate, 
                          l1_regularizer, l2_regularizer, 
                          lstm_layers,dense_layers)
        ut.save_results(filenamepath, tobalanceMinority, sampling_over, 0, res)
    
        
        
# e = int(sys.argv[1])
# b = int(sys.argv[2])
# s = int(sys.argv[3])
# h = int(sys.argv[4])
# d = float(sys.argv[5])
# lr = float(sys.argv[6])
# l1 = float(sys.argv[7])
# l2 = float(sys.argv[8])
# lstmi = int(sys.argv[9])
# densei = int(sys.argv[10])
#use_cdisum = int(sys.argv[11])
e=100
b=100
s=3
h=12
d=.2
lr=.01
l1=.01
l2=.1
lstmi=1
densei=1
use_cdisum=True
n_splits = 5
do_deviations = True
do_normalize = True

for indata in range(3):   
    do_analysis(use_cdisum, do_deviations, do_normalize, n_splits, False, False, indata, e, b, d, lr, l1, l2, lstmi, densei, s, h)
  
        
