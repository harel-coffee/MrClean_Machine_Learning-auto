# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:15:31 2017

@author: laramos
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
#from fancyimpute import MICE

"""
These methods below only select the variables per study.

"""
def Change_One_Hot(frame,vals_mask):
    """
    This function one-hot-encode the features from the vals_mask and returns it as numpy array
    Input:
        frame: original frame with variables
        vals_mask: array of string with the names of the features to be one-hot-encoded [['age','sex']]
    Ouput:
        Result: One-hot-encoded feature set in pd.frame format
    """
    new_frame=frame[vals_mask]
    X_vars=np.array(new_frame,dtype='float64')
    rf_enc = OneHotEncoder()
    rf_enc.fit(X_vars)
    Result=rf_enc.transform(X_vars)
    Result=Result.toarray()
    Result=np.array(Result,dtype='float64')
   
    feat_ind=np.zeros(X_vars.shape[1],dtype= 'int16')
    for i in range(0,X_vars.shape[1]):
        feat_ind[i]=(np.unique(X_vars[:,i]).shape[0])

    
    cols=list()    
    for i in range(0,feat_ind.shape[0]):
        for j in range(0,feat_ind[i]):
            cols.append(vals_mask[i]+str(j))
    
    Result=pd.DataFrame(Result,columns=cols)   
        
    return(Result,cols)
        
def Normalize_Min_Max(x):
    
    max_val=np.max(x,axis=0)
    min_val=np.min(x,axis=0)
    for i in range(x.shape[1]):
        x[:,i]=(x[:,i]-min_val[i])/(max_val[i]-min_val[i])
    return(x)
 

def Encode_Variables(X,cols,vals_mask):
    new_frame=pd.DataFrame(X,columns=cols)
        
    cat_feats,cols_onehot=Change_One_Hot(new_frame,vals_mask)
    
    final_frame=pd.concat([new_frame,cat_feats],axis=1)
    
    #here we drop the variables that we want to change for onehot representations available in cat_feats
    final_frame=final_frame.drop(vals_mask,axis=1)
    
    cols=list(final_frame.columns)
    
    new_X=np.array(final_frame,dtype='float64')
    
    return(new_X,cols)
    
def Split_Center(center,cnonr):
    
    train=center!=cnonr
    test=center==cnonr

    return(train,test)

def Combine_Center5_10(center):
    """
    this fuction combines center 5 with center 10 by repalcing center 5 entries.
    After that we have to skip center 5 in the for loop, so a range of values is created (range_centers) and cente5 is deleted.
    
    """
    pos5=np.where(center==5)
    center[pos5]==10
    range_centers=np.arange(17)
    range_centers=np.delete(range_centers,5,axis=0)
    return(center,range_centers)
    
    

def Get_Vars_Baseline_binscores(frame,path_variables):  
    var=pd.read_csv(path_variables)
    
    for i in range(0,len(var)):
        var.iloc[i]['names']=str(var.iloc[i]['names']).replace(" ","")
    frame=frame[var['names']]

    vals_mask=['premrs','collaterals','smoking','occlsegment_c','cbs_occlsegment_recoded']

    return(frame,vals_mask)


def Get_Vars_Baseline_contscores(frame,path_variables):  
    var=pd.read_csv(path_variables)
    
    for i in range(0,len(var)):
        var.iloc[i]['names']=str(var.iloc[i]['names']).replace(" ","")
    frame=frame[var['names']]
    
    vals_mask=['premrs','collaterals','smoking','occlsegment_c','cbs_occlsegment_recoded']
    
    return(frame,vals_mask)

#dichotomise
#'premrs','collaterals','pretici_c','occlsegment_c','occlsegmangio_c','smoking'

 
def Get_Vars_All_binscores(frame,path_variables):   
    
    var=pd.read_csv(path_variables)
    for i in range(0,len(var)):
        var.iloc[i]['names']=str(var.iloc[i]['names']).replace(" ","")
    frame=frame[var['names']]

    vals_mask=['premrs','pretici_c','posttici_c','preaol_c','postaol_c','collaterals','smoking','occlsegmangio_c','performedproc','iatreatment1','occlsegment_c','cbs_occlsegment_recoded']

    return(frame,vals_mask)    
#dichotomise
#'premrs','collaterals','pretici_c','occlsegment_c','occlsegmangio_c','smoking','performedproc','disloc'

 #This was created to add postici and cluster all adverse events into any     
def Get_Vars_All_contscores(frame,path_variables):  
    frame=frame.drop('posttici_c',axis=1)
    var=pd.read_csv(path_variables)
    for i in range(0,len(var)):
        var.iloc[i]['names']=str(var.iloc[i]['names']).replace(" ","")
    frame=frame[var['names']]

    #vals_mask=['premrs','pretici_c','posttici_c','preaol_c','postaol_c','collaterals','smoking','occlsegmangio_c','performedproc','iatreatment1','occlsegment_c','cbs_occlsegment_recoded']
    #below no postitici, for sensitivity analysis after rebutal
    vals_mask=['premrs','pretici_c','preaol_c','postaol_c','collaterals','smoking','occlsegmangio_c','performedproc','iatreatment1','occlsegment_c','cbs_occlsegment_recoded']
    print("Got all right variables")
    return(frame,vals_mask)  
#dichotomise
#'premrs','collaterals','pretici_c','occlsegment_c','occlsegmangio_c','smoking','performedproc','disloc'



def Get_Vars_priorknowledge_baseline(frame,path_variables):  
    var=pd.read_csv(path_variables)
    for i in range(0,len(var)):
        var.iloc[i]['names']=str(var.iloc[i]['names']).replace(" ","")
    frame=frame[var['names']]
    vals_mask=['premrs','collaterals','occlsegment_c']
    return(frame,vals_mask)
#dichotomise
#'premrs','collaterals','pretici_c','occlsegmangio_c','smoking','collaterals_ex'

def Get_Vars_priorknowledge_all(frame,path_variables): 
    var=pd.read_csv(path_variables)
    for i in range(0,len(var)):
        var.iloc[i]['names']=str(var.iloc[i]['names']).replace(" ","")
    frame=frame[var['names']]
    #vals_mask=['premrs','collaterals','occlsegmangio_c','posttici_c']
    vals_mask=['premrs','collaterals','occlsegmangio_c']
    return(frame,vals_mask)
#dichotomise
#'premrs','collaterals','pretici_c','occlsegment_c_short','smoking','collaterals_ex'


    
def Change2_Missing_spss(data,cols): 
    """
    In the spss file many features have different values for missing, like 2 instead of np.nan, here we change those
    Input = frame with wrong missing values
    Output = Fixed frame
    
    """
    cols=pd.Index.tolist(cols)
    pos=(int)(cols.index('ivtrom'))
    
    print(data.shape)
    print(pos)
    print(data[0,0]>2)
    for i in range(0,data.shape[0]):
        if data[i,pos]>=2:
            data=np.nan
    return(data)   
    
    
def Impute_and_Save(f):
    raw_data_list = list(f) 
    frame = pd.DataFrame(raw_data_list) 
    frame = frame.rename(columns=frame.loc[0]).iloc[1:] 
    #frame=frame.drop([b'StudySubjectID'],axis=1)
    cols=frame.columns #these columns are in a binary format, below they are converted to string
    colsaux=[]
    
    for i in range(0,cols.shape[0]):
        colsaux.append(cols[i].decode('UTF8'))

    cols=colsaux    
    frame.columns=cols
    
    arr=np.array(frame['StudySubjectID'])
    for i in range(0,arr.shape[0]):
        arr[i]=arr[i].decode('UTF8')
    np.save('E:\\Adam\\sub_id_complete.npy',arr)
    
    frame.fillna(value=np.nan,inplace=True)
    
    Ys=(frame[['mrs','posttici_c']]).values
    #Ys=(frame[['mrs']]).values
    cont_mis=0
    for i in range(0,frame.shape[0]):
        if (np.isnan(Ys[i,0]) or np.isnan(Ys[i,1])):
            print('missing:', arr[i])
            frame=frame.drop([i+1])
            cont_mis=cont_mis+1 
            
    print(cont_mis)
    arr=np.array(frame['StudySubjectID'])
    for i in range(0,arr.shape[0]):
        arr[i]=arr[i].decode('UTF8')
    np.save('E:\\Adam\\sub_id.npy',arr)
    return(arr)
    
    
    #cols=frame.columns
    Ys_final=(frame[['mrs','posttici_c']]).values
    
    frame=frame.drop(['mrs','posttici_c'],axis=1)
    
    cols=frame.columns
    for i in range (0,frame.shape[1]):
        col=frame.columns[i]
        val=frame[col]
        s=val.dtype
        if s=='object':
            frame[col]=pd.to_numeric(frame[col],errors='coerce')
          
    
    dataread=np.array(frame.values,dtype='float64')    
    num_miss=np.zeros(dataread.shape[1])
    
    for i in range(0,dataread.shape[0]):
        for j in range(0,dataread.shape[1]):
            if np.isnan(dataread[i,j]):
                num_miss[j]=num_miss[j]+1
    #how much% is missing, 25% or more
    for i in range(0,dataread.shape[1]):
        num_miss[i]=(num_miss[i]*100)/dataread.shape[0]
        
    #check if below <25 because I want the above 25% to be 0 and eliminated 

    
    cols_delete=num_miss<25
    for i in range(0,len(cols)):
        if cols_delete[i]==1:
            print(cols[i],num_miss[i])
        
    dataread=dataread[:,cols_delete]
    
    
    cols=cols[cols_delete]    
   
    mice=MICE()
    X=mice.complete(dataread)
    cols_ind=pd.Index.tolist(cols)

    
    #df=pd.DataFrame(X,columns=cols_ind)
    
    df=pd.DataFrame(X)
    
    df=df.round()
    df_new=df
    
    cols_ind.append('mrs')
    cols_ind.append('posttici_c')
    #df_new=pd.concat([df_new,Ys_final],axis=1)
    
    X=np.array(df_new,dtype='float64') 
    X=np.concatenate((X,Ys_final),axis=1)
    
    df_new=pd.DataFrame(X,columns=cols_ind)
    
    return(X,cols,df_new)
    
 
def Fix_Dataset_spss(f,label_name,feats_use,binary_mrs):
    """
    This function reads the dataset in a spss format, selected only the important collumns,
    preprocess a few of them into cathegories and performs imputation using random forests or MICE
    Input:
    f = returned from spss.SavReader(filename)
    label_name = name of the column to be predicted, the label for the features (Y)
    Feats_use = This parameter specifies which variables will be selected ('Baseline_Imp','Baseline_NonImp','ALL_NonImp')
    binary_mrs= If true returns a binary version of mrs >2 =1 and <=2 =0, if false returns it from 1 to 6 (multiclass)
    Output:
        X = Dataset features with imputed values (mxn)
        Y = Labels (m)
        cols = columns names so one can trace back each feature

    """

                
    raw_data_list = list(f) 
    frame = pd.DataFrame(raw_data_list) 
    frame = frame.rename(columns=frame.loc[0]).iloc[1:] 
    original_frame=frame
    cols=frame.columns #these columns are in a binary format, below they are converted to string
    colsaux=[]
    
    for i in range(0,cols.shape[0]):
        colsaux.append(cols[i].decode('UTF8'))
    cols=colsaux    
    frame.columns=cols


    frame.fillna(value=np.nan,inplace=True)
    
    #Patients with missing mrs or postici are deleted    
    Ys=(frame[['mrs','posttici_c']]).values    
    cont_mis=0
    for i in range(0,frame.shape[0]):
        if (np.isnan(Ys[i,0]) or np.isnan(Ys[i,1])):
            #print("deleting ",Ys_t.loc[i+1,['mrs']],Ys_t.loc[i+1,['posttici_c']])
            frame=frame.drop([i+1])
            cont_mis=cont_mis+1 
    

    Y=frame[label_name]
    Y=np.array(Y,dtype='int32')

    print(cont_mis)
    if binary_mrs:
        for i in range(0,Y.shape[0]):
            if Y[i]>2:
                Y[i]=1
            else:
                Y[i]=0
    
    frame=frame.drop(label_name,axis=1)  

    #checking what kind of features will be used based on the experiment set up
    frame,vals_mask = {
          'Baseline_binscore': lambda frame:Get_Vars_Baseline_binscores(frame,path_variables),
          'Baseline_contscore': lambda frame:Get_Vars_Baseline_contscores(frame,path_variables),
          'All_vars_binscore':lambda frame:Get_Vars_All_binscores(frame,path_variables),
          'All_contscore': lambda frame:Get_Vars_All_contscores(frame,path_variables),
          'Knowledge_baseline': lambda frame:Get_Vars_priorknowledge_baseline(frame,path_variables),
          'Knowledge_all': lambda frame:Get_Vars_priorknowledge_all(frame,path_variables),
    }[feats_use](frame)
    
    for i in range (0,frame.shape[1]):
        col=frame.columns[i]
        val=frame[col]
        s=val.dtype
        if s=='object':
            frame[col]=pd.to_numeric(frame[col],errors='coerce')
        
       
    cols=frame.columns
    
    dataread=np.array(frame.values,dtype='float64')    
    num_miss=np.zeros(dataread.shape[1])
    
    for i in range(0,dataread.shape[0]):
        for j in range(0,dataread.shape[1]):
            if np.isnan(dataread[i,j]):
                num_miss[j]=num_miss[j]+1
    #how much% is missing, 25% or more
    for i in range(0,dataread.shape[1]):
        num_miss[i]=(num_miss[i]*100)/dataread.shape[0]
        
    #check if below <25 because I want the above 25% to be 0 and eliminated 
    cols_delete=num_miss<25
    
    #for j in range(0,dataread.shape[1]):
    #    if num_miss[j]>1000:
    #        print(cols[j])
    
    dataread=dataread[:,cols_delete]
    cols=cols[cols_delete]
    #X=imp.IARI(dataread,Y)

    #frame=Change2_Missing_spss(dataread,cols)        
    
    
    mice=MICE()
    X=mice.complete(dataread)

    df=pd.DataFrame(X,columns=cols)
    
    df=df.round()
    df_new=df
    
    
    X=np.array(df_new,dtype='float64') 
    
    return(X,Y,cols,original_frame)

def Fix_Dataset_csv(path_data,label_name,feats_use,path_variables):
    """
    This function reads the dataset in a spss format, selected only the important collumns,
    preprocess a few of them into cathegories and performs imputation using random forests or MICE
    Input:
    
    label_name = name of the column to be predicted, the label for the features (Y)
    Feats_use = This parameter specifies which variables will be selected ('Baseline_Imp','Baseline_NonImp','ALL_NonImp')
    binary_mrs= If true returns a binary version of mrs >2 =1 and <=2 =0, if false returns it from 1 to 6 (multiclass)
    Output:
        X = Dataset features with imputed values (mxn)
        Y = Labels (m)
        cols = columns names so one can trace back each feature

    """

    frame=pd.read_csv(path_data)
    

    
    #frame=frame[frame.posttici_c>=3]
    #frame=frame[frame.posttici_c<1]
    
    #frame=frame.drop('posttici_c_bin',axis=1)
    
   
    cols=frame.columns #these columns are in a binary format, below they are converted to string
    center=frame['cnonr'].values       
    Y=frame[label_name]
    Y=np.array(Y,dtype='int32')

    """
    for i in range(0,Y.shape[0]):
            if Y[i]>2:
                Y[i]=1
            else:
                Y[i]=0
    """
    frame=frame.drop(label_name,axis=1)  
               
          
    
    #checking what kind of features will be used based on the experiment set up
    frame,vals_mask = {
          'Baseline_binscore': lambda frame:Get_Vars_Baseline_binscores(frame,path_variables),
          'Baseline_contscore': lambda frame:Get_Vars_Baseline_contscores(frame,path_variables),
          'All_vars_binscore':lambda frame:Get_Vars_All_binscores(frame,path_variables),
          'All_contscore': lambda frame:Get_Vars_All_contscores(frame,path_variables),
          'Knowledge_baseline': lambda frame:Get_Vars_priorknowledge_baseline(frame,path_variables),
          'Knowledge_all': lambda frame:Get_Vars_priorknowledge_all(frame,path_variables),
    }[feats_use](frame)
    
    
    
    cols=frame.columns
    
    
    dataread=np.array(frame)    
    
    np.isnan(dataread.any())
      
    
    return(dataread,Y,cols,center,vals_mask)
    
def Fix_Dataset_Core_dta(f,label_name,feats_use):
    """
    This function reads the dataset in a spss format, selected only the important collumns,
    preprocess a few of them into cathegories and performs imputation using random forests or MICE
    Input:
    f = returned from spss.SavReader(filename)
    label_name = name of the column to be predicted, the label for the features (Y)
    Feats_use = This parameter specifies which variables will be selected ('Baseline_Imp','Baseline_NonImp','ALL_NonImp')
    binary_mrs= If true returns a binary version of mrs >2 =1 and <=2 =0, if false returns it from 1 to 6 (multiclass)
    Output:
        X = Dataset features with imputed values (mxn)
        Y = Labels (m)
        cols = columns names so one can trace back each feature

    """

    path='E:\\Mrclean\\Data\\'
    path_variables='E:\\Mrclean\\Data\\Variables\\'


    #feats_use='Baseline_contscore'
    feats_use='All_contscore'
    path_variables=path_variables+feats_use+".csv"
    frame=pd.io.stata.read_stata((path+"RegistryOpenclinicacheck_core.dta"))

    #delete patients with missing mrs
    Y_mrs=(frame[['mrs']]).values    
    cont_mis=0
    
    Y_tici=(frame['posttici_c']).values    
    cont_mis_tici=0
    Y_tici=frame['posttici_c'].factorize([np.nan,'0','1','2A','2B','2C','3'])[0]
           
    to_delete=list()
    for i in range(0,frame.shape[0]):
        if (np.isnan(Y_mrs[i]) or (Y_tici[i]==0)):
            to_delete.append(i+1)
            frame=frame.drop([i])
            cont_mis=cont_mis+1 
            cont_mis_tici=cont_mis_tici+1 
              
          
    Y_mrs=frame['mrs']
    Y_mrs=np.array(Y_mrs,dtype='int32')
    Y_tici=frame['posttici_c'].values
    Y_tici=frame['posttici_c'].factorize(['0','1','2A','2B','2C','3'])[0]  
    cnonr=frame['cnonr']
    #checking what kind of features will be used based on the experiment set up
    frame,vals_mask = {
          'Baseline_binscore': lambda frame:Get_Vars_Baseline_binscores(frame,path_variables),
          'Baseline_contscore': lambda frame:Get_Vars_Baseline_contscores(frame,path_variables),
          'All_vars_binscore':lambda frame:Get_Vars_All_binscores(frame,path_variables),
          'All_contscore': lambda frame:Get_Vars_All_contscores(frame,path_variables),
          'Knowledge_baseline': lambda frame:Get_Vars_priorknowledge_baseline(frame,path_variables),
          'Knowledge_all': lambda frame:Get_Vars_priorknowledge_all(frame,path_variables),
    }[feats_use](frame)
    frame=pd.concat([frame,cnonr],axis=1)        
    cols=frame.columns
    for i in range(0,frame.shape[1]):
        if frame[cols[i]].dtype.name=='category':
            cat=frame[cols[i]].cat.categories
            print(frame[cols[i]].isnull().values.any(),cols[i])
            frame[cols[i]],labels=frame[cols[i]].factorize([np.nan,cat])
            if len(cat)<len(labels):
                frame[cols[i]]=frame[cols[i]].replace(0,np.nan)
                frame[cols[i]]=frame[cols[i]]-1

    frame.fillna(value=np.nan,inplace=True)    


    
    print(cont_mis)
        #poor outcome =1
    for i in range(0,Y_mrs.shape[0]):
        if Y_mrs[i]>2:
                Y_mrs[i]=1
        else:
                Y_mrs[i]=0

    #poor perfusion (1) <3 (0 1 2 3 4 5)(0  1 2a 2b 2c 3)
    for i in range(0,Y_tici.shape[0]):
        if Y_tici[i]<3:
                Y_tici[i]=1
        else:
                Y_tici[i]=0
    #frame=frame.drop('mrs',axis=1)  
    #frame=frame.drop('posttici_c',axis=1)  

    
    for i in range (0,frame.shape[1]):
        col=frame.columns[i]
        val=frame[col]
        s=val.dtype
        if s=='object':
            frame[col]=pd.to_numeric(frame[col],errors='coerce')
        
       
    cols=frame.columns
    
    dataread=np.array(frame.values,dtype='float64')    
    num_miss=np.zeros(dataread.shape[1])
    
    for i in range(0,dataread.shape[0]):
        for j in range(0,dataread.shape[1]):
            if np.isnan(dataread[i,j]):
                num_miss[j]=num_miss[j]+1
    #how much% is missing, 25% or more
    for i in range(0,dataread.shape[1]):
        num_miss[i]=(num_miss[i]*100)/dataread.shape[0]
        
    #check if below <25 because I want the above 25% to be 0 and eliminated 
    cols_delete=num_miss<25
    
    #for j in range(0,dataread.shape[1]):
    #    if num_miss[j]>1000:
    #        print(cols[j])
    
    dataread=dataread[:,cols_delete]
    cols=cols[cols_delete]
    #X=imp.IARI(dataread,Y)

    #frame=Change2_Missing_spss(dataread,cols)        
    
    
    mice=MICE()
    X=mice.complete(dataread)

    df=pd.DataFrame(X,columns=cols)
    
    df=df.round()
    df_new=df
    
    Y_tici=pd.DataFrame(Y_tici,columns=['posttici_c'])
    Y_mrs=pd.DataFrame(Y_mrs,columns=['mrs'])
    frame=pd.concat([df_new,Y_tici,Y_mrs],axis=1)
    #frame.to_csv(r'//home//user//Desktop//Codes//Codes//Baseline_Data.csv')
    frame.to_csv(r'//home//user//Desktop//Codes//Codes//AllVariables_Data.csv')
    
    
    return(X,Y,cols,original_frame)

  