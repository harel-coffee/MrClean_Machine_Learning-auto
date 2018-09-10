# -*- coding: utf-8 -*-
"""
This Code generates odds ratio,p-value and confidence interval for Mrclean registry data

Important: The name of the features is important, thats how they are dichotomized.

For more info about dichotomization procedure check the actual .csv file



@author: laramos
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score,auc
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge
from scipy import interp
import math
from sklearn.externals import joblib
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import logistic
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import savReaderWriter as spss
import Methods as mt
import Data_Preprocessing as pp
import Feature_Selection as fs
#path_data ='E:\\Mrclean\\Data\\RAWComplete_Imputed_Dataset.csv' 
path_data ='E:\\Mrclean\\Data\\AllVariables_Data.csv' 
#path_data ='E:\\Mrclean\\Data\\Baseline_Data.csv' 

path_variables='E:\\Mrclean\\Data\\Variables\\'


#feats_use='Baseline_contscore'
feats_use='All_contscore' #Done
#feats_use='Knowledge_baseline'
#feats_use='Knowledge_all'
 

label_use='mrs'
#label_use='posttici_c'

#SELECT FEATURES
select_feats_logit=True

method='rfc'
T=0.01

#method='lasso'
#T=0.1

#method='elastik'  
#T=0.01

#method='backward'  

path_variables=path_variables+feats_use+".csv"
   
[X,Y,cols,center,vals_mask]=pp.Fix_Dataset_csv(path_data,label_use,feats_use,path_variables)
#data=pd.io.stata.read_stata(("E:\\Mrclean\\Data\\RegistryOpenclinicacheck_core.dta"))
center,range_centers=pp.Combine_Center5_10(center)

[X,cols]=pp.Encode_Variables(X,cols,vals_mask)     
Y2=np.expand_dims(Y,axis=1)
cols.append('mrs')
X2=np.concatenate((X,Y2),axis=1)
splits=16
frame=pd.DataFrame(X2,columns=cols)
#frame=pd.get_dummies(frame)
#dum=pd.get_dummies(frame[cols[0]], prefix = (cols[0]+'nl')) 
#frame=frame.drop(cols[0],axis=1)  
#frame = pd.concat([frame, dum], axis=1)

aucs=np.zeros(splits)
aucs2=np.zeros(splits)
s='mrs ~ '

for i in range(0,frame.shape[1]-1):

        s=s+ cols[i]
        if i<78:
            s=s+"+"
odds=0
oddsLR=0
for l in range(splits):                    
        train,test=pp.Split_Center(center,range_centers[l])            
        frame_train=frame[train]
        frame_test=frame[test]   
        
        #logit = smf.glm(formula=s, data=frame_train, family=sm.families.Binomial())
        
        #logit = sm.Logit(Y[train],X[train])
        #res = logit.fit(method='bfgs')
        #print(res.mle_retvals)
        #res=logit.fit_regularized()
        clf=LogisticRegression()
        #clf=RandomForestClassifier()
        clf.fit(X[train],Y[train])
        
        prob=clf.predict_proba(X[test])[:,1]
        
        #pred=res.predict(X[test])
        
        #aucs[l]=roc_auc_score(frame_test['mrs'],pred)
        
        aucs2[l]=roc_auc_score(frame_test['mrs'],prob)
        
        #odds=np.exp(-res.params)
        oddsLR=oddsLR+np.exp(clf.coef_)

cols.pop()
cols=pd.Index(cols)
oddsLR=oddsLR/splits
oddsLR=oddsLR.reshape(-1)
data = pd.Series(oddsLR, index=cols)
#data=data.sort_values()
#data.to_csv("Baseline_Postici_LR.csv")
data.to_csv("ALL_MRS_LR.csv")

"""
pvals=res.pvalues

conf=np.exp(-res.conf_int())

print(cols[i])
#print(res.summary())        print("pvalues = ",pvals[1])
print("odds ratio = ",odds[1])
for j in range(1,conf.index.shape[0]):
 print("confidence interval = ",conf.index[j],conf.loc[conf.index[j],1], conf.loc[conf.index[j],0])
print('\n ')
"""
      