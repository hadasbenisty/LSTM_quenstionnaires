# -*- coding: utf-8 -*-
"""
Created on Sun May 29 09:58:52 2022

@author: hadas
"""
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler




class w_data():    
    def __init__(self, use_cdisum, do_deviations, do_normalize, filenamepath, inputkeyList, outputkey, tobalanceMinority, sampling_over, n_splits = 5, max_day_length = 30):
        
        self.n_splits = n_splits
        data = pd.read_csv(filenamepath, delimiter=',')
        
         
        
        self.outputkey = outputkey
        
        self.trainingList = []
        self.testingList = []        
        self.tobalanceMinority = tobalanceMinority
        self.sampling_over = sampling_over
        # remove participants with zero pos+neg interactions with mom/dad
        
        posdad_input = self.get_mean_by_participant(data["pos_inter_dad_sum"], data["Participant"])
        negdad_input = self.get_mean_by_participant(data["neg_inter_dad_sum"], data["Participant"])
        
        posmom_input = self.get_mean_by_participant(data["pos_inter_mom_sum"], data["Participant"])
        negmom_input = self.get_mean_by_participant(data["neg_inter_mom_sum"], data["Participant"])
        nonzeroloc1 = (posdad_input + negdad_input) != 0  
        nonzeroloc2 = (posmom_input + negmom_input) != 0 
        nonzeroloc = nonzeroloc1*nonzeroloc2
        newdata = {}
        for k in data.keys():
            v = data[k][np.where(nonzeroloc)[0]]
            newdata[k] = v
        data = newdata
        self.participantsList = np.unique(data["Participant"])
        self.get_train_test_indices()
        data[outputkey] = np.double(data[outputkey])
        for k in inputkeyList:
            data[k] = np.double(data[k])
        ## add mean level of outcome variable
        mean_output = self.get_mean_by_participant(data[outputkey], data["Participant"])
        data[outputkey + "_mean"] = mean_output
        final_inputKeyList = []
        if use_cdisum:
            final_inputKeyList.append(outputkey + "_mean")
        
        for k in inputkeyList:
            mean_input = self.get_mean_by_participant(data[k], data["Participant"])
            data[k + "_mean"] = mean_input
            data[k + "_dev"] = np.double(data[k]) - mean_input
            final_inputKeyList.append(k + "_mean")
            final_inputKeyList.append(k + "_dev")
        
        
        if do_deviations:
            self.inputkeyList = final_inputKeyList    
        else:
            self.inputkeyList = inputkeyList
        if do_normalize:
            data[self.outputkey] = data[self.outputkey] - np.mean(data[self.outputkey])
            data[self.outputkey] = data[self.outputkey]/np.std(data[self.outputkey])
            for key in self.inputkeyList:
                data[key] = data[key] - np.mean(data[key])
                data[key] = data[key]/np.std(data[key])
        self.data = data

    def get_mean_by_participant(self, x, Participant):
        
        mean_output = np.double(np.zeros_like(x))
        unique_list = np.unique(Participant)
        
        for p in unique_list:
            mean_val = np.mean(x[Participant == p])
            mean_output[Participant == p] = mean_val
        return mean_output
    def get_single_feat(self, participantsList, k):
        a = self.data[k]  
        y = []  
      
        for p in participantsList:
          mycurrentdata = a[self.data["Participant"]==p].tolist() 
          if np.any(mycurrentdata == np.NAN):
              print("hi")
          y.append(mycurrentdata)  
      
        #flattening our list
        y = [item for sublist in y for item in sublist]
        #converting to 2D numpy array
        y = np.array(y).reshape(-1,1).ravel()  
        return y
    def make_data_by_list(self, ParticipantsList):
        newdata = {}
        newdata[self.outputkey] = self.get_single_feat(ParticipantsList, self.outputkey)
        for k in self.inputkeyList:
            newdata[k] = self.get_single_feat(ParticipantsList, k)
        newdata["Participant"] = self.get_single_feat(ParticipantsList, "Participant")
        newdata["Dyad.x"] = self.get_single_feat(ParticipantsList, "Dyad.x")
        
        
        return newdata
    def load_XY_data(self, listinds, sequence_length): 
       
        X = self.load_X_data_many2one(listinds, sequence_length)
        y = self.load_Y_data_many2one(listinds, sequence_length)
       
        if self.tobalanceMinority:
            if self.sampling_over:
                ros = RandomOverSampler(random_state=42, sampling_strategy='minority')
            else:
                ros = RandomUnderSampler(random_state=42)
            X, y = ros.fit_resample(X, y)
        return X, y 
   
   
    def load_X_data_many2one(self, participantsList, sequence_length):  
      finalarray = []  
      for p in participantsList:
        participant_days = sum(self.data['Participant']==p)
        for i in range(participant_days-sequence_length+1):
          participantdata = []
          for key in self.inputkeyList:
            a = self.data[key]  
            mycurrentdata = a[self.data['Participant']==p].tolist()
            mycurrentdata = mycurrentdata[i:i+sequence_length]
            participantdata.extend(mycurrentdata)
          
          participantdata = np.array(participantdata).reshape(len(self.inputkeyList), sequence_length)
          participantdata = np.transpose(participantdata)
          #print(str(i) + " " +str(p))
          if ((i == 0) and (p == participantsList[0])):
              finalarray = participantdata
          elif len(finalarray.shape) == 2:
              finalarray = np.stack((finalarray, participantdata))
          else:
              finalarray = np.vstack((finalarray, participantdata[None]))
      return finalarray
   
    def load_Y_data_many2one(self, participantsList, sequence_length):
      a = self.data[self.outputkey]  
      finalarray = []  
    
      for p in participantsList:
        mycurrentdata = a[self.data["Participant"]==p].tolist()
        del mycurrentdata[0:sequence_length-1] #deleting first [sequence_length] observations
        finalarray.append(mycurrentdata)  
    
      #flattening our list
      finalarray = [item for sublist in finalarray for item in sublist]
      #converting to 2D numpy array
      finalarray = np.array(finalarray).reshape(-1,1).ravel()  
    
      return finalarray
    
        


    def get_train_test_indices(self):
      kfold0 = KFold(n_splits = self.n_splits)
      kfold = KFold(n_splits = self.n_splits)
      trainingList = []
      trainingList0 = []
      devList = []
      testingList = []
      for train_index0, test_index in kfold0.split(self.participantsList):
          testingList.append(self.participantsList[test_index])
          trainingList0.append(self.participantsList[train_index0])
          for train_index, dev_index in kfold.split(self.participantsList[train_index0]):
              trainingList.append(self.participantsList[train_index0[train_index]])
              devList.append(self.participantsList[train_index0[dev_index]])
      self.trainingList = trainingList
      self.devList = devList
      self.testingList = testingList
      self.trainingList0 = trainingList0

