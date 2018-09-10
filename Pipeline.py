
"""
This code is for the EEg project, it contains feature extraction and data pre-processing

The filter fit in python is different from matlab, it is giving me different values from marjolein
@author: laramos
"""
import numpy as np
import os

#os.chdir(os.getcwd())
os.chdir('E:\Mrclean\Code')

import glob
import Methods as mt
import Data_Preprocessing as dp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score,confusion_matrix,brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import welch
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
from scipy import interp  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
from sklearn.metrics import auc
from random import shuffle
import time
import pandas as pd
import Data_Preprocessing as pp
import Feature_Selection as fs

import importlib.util
importlib.reload(pp)

class Measures:    
    
    def __init__(self, splits,num_feats):
        self.clf_auc=np.zeros(splits)
        self.clf_brier=np.zeros(splits)
        self.clf_sens=np.zeros(splits)
        self.clf_spec=np.zeros(splits)
        self.mean_tpr=0.0
        self.frac_pos_rfc=np.zeros(splits)
        self.run=False
        self.feat_imp=np.zeros((splits,num_feats))

if __name__ == '__main__':
    
    #path_data ='E:\\Mrclean\\Data\\RAWComplete_Imputed_Dataset.csv' 
    #path_data ='E:\\Mrclean\\Data\\AllVariables_Data.csv' 
    path_data ='E:\\Mrclean\\Data\\Baseline_Data.csv' 
    
    path_variables='E:\\Mrclean\\Data\\Variables\\'


    feats_use='Baseline_contscore'    
    #feats_use='All_contscore' 
    #feats_use='Knowledge_baseline'
    #feats_use='Knowledge_all'
    
 
    
    label_use='mrs'
    #label_use='posttici_c'
    
    #SELECT FEATURES
    select_feats_logit=False
    
    #method='rfc'
    #T=0.01 #mrs
    #T=0.05 #pt
    
    method='lasso'
    T=0.01
    #T=0.00000000
    
    #method='elastik'  
    #T=0.01
    #T=0.00
    
    #method='backward'  
    #T=0
    
    path_variables=path_variables+feats_use+".csv"
    #path_variables=path_variables+feats_use+".csv"
    path_results='E:\\Mrclean\\Results_Sens\\test'+'-'+feats_use+'-'+label_use+'-'+method+'\\'
    path_models=path_results+"\\Models"
    if not os.path.exists(path_results):
         os.makedirs(path_results)
         os.makedirs(path_models)
    
   
    [X,Y,cols,center,vals_mask]=pp.Fix_Dataset_csv(path_data,label_use,feats_use,path_variables)
    #data=pd.io.stata.read_stata(("E:\\Mrclean\\Data\\RegistryOpenclinicacheck_core.dta"))
    center,range_centers=pp.Combine_Center5_10(center)

    [X,cols]=pp.Encode_Variables(X,cols,vals_mask)   


    cols=np.array(cols)
    np.save(path_results+'cols.npy',cols)
    
    
    #X=pp.Normalize_Min_Max(X)
    
    num_feats=X.shape[1]
    splits=100
    cv=5
    
    mean_tprr = 0.0
    
    rfc_m=Measures(splits,num_feats)
    svm_m=Measures(splits,num_feats)
    lr_m=Measures(splits,num_feats)
    nn_m=Measures(splits,num_feats)
    sl_m=Measures(splits,num_feats)    
    #thresholds = np.linspace(0.00001, 0.1, num=10)
    
    start_pipeline = time.time()
    
    skf=StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=0)
    
    #for l in range(splits):
    for l, (train, test) in enumerate(skf.split(X, Y)):
        print("----------------Iteration:----------------",l)
                    
        #train,test=pp.Split_Center(center,range_centers[l])
        
        x_train=X[train]
        x_test=X[test]
        y_train=Y[train]
        y_test=Y[test]

        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train=scaler.transform(x_train)
        x_test=scaler.transform(x_test) 
                
        class_rfc=mt.RFC_Pipeline(False,x_train,y_train,x_test,y_test,l,cv,mean_tprr,rfc_m,path_results)   
        class_svm=mt.SVM_Pipeline(False,x_train,y_train,x_test,y_test,l,svm_m,cv,mean_tprr,path_results)   
        class_lr=mt.LR_Pipeline(True,x_train,y_train,x_test,y_test,l,mean_tprr,lr_m,cv,select_feats_logit,T,method) 
        class_nn=mt.NN_Pipeline(False,x_train,y_train,x_test,y_test,l,nn_m,cv,mean_tprr,path_results) 
        class_sl=mt.SL_Pipeline(False,x_train,y_train,x_test,y_test,l,sl_m,mean_tprr,class_rfc,class_svm,class_lr,class_nn)   
        

    end_pipeline = time.time()    
    print("Total time to process: ",end_pipeline-start_pipeline)
    final_m=[rfc_m,svm_m,lr_m,nn_m,sl_m]
    final_m=[x for x in final_m if x.run != False]
    names=[class_rfc.name,class_svm.name,class_lr.name,class_nn.name,class_sl.name]
    names=[x for x in names if x != 'NONE'] 
    mt.Print_Results_Excel(final_m,splits,names,path_results)
    
