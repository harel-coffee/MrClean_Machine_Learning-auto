
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV                               
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import brier_score_loss
import random as rand
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
import time
from scipy.stats import randint as sp_randint
import scipy as sp
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy import interp 
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFECV
from sklearn.naive_bayes import GaussianNB
import xlwt
import superlearner as sl

n_jobs=5
class RFC_Pipeline: 
 
    def RandomGridSearchRFC(self,X,Y,splits,path_results):
        """
        This function looks for the best set o parameters for RFC method
        Input: 
            X: training set
            Y: labels of training set
            splits: cross validation splits, used to make sure the parameters are stable
        Output:
            clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
        """    
        
        start_rfc = time.time()  
        
        tuned_parameters = {
        'n_estimators': ([200,400,500,600,800,1000,1200,1400,1600,1800,2000]),
        'max_features': (['auto', 'sqrt', 'log2']),                   # precomputed,'poly', 'sigmoid'
        'max_depth':    ([10,20,30,40, 50, 60, 70, 80, 90, 100, None]),
        'criterion':    (['gini', 'entropy']),
        'min_samples_split':  [2,4,6,8],
        'min_samples_leaf':   [2,4,6,8,10]
        }
        
        scores = ['roc_auc']   

        random_state=np.random.randint(0,1000)
        rfc = RandomForestClassifier(n_estimators=25, oob_score = True,random_state=random_state)   
        
        print("RFC Grid Search")
        clf =  RandomizedSearchCV(rfc, tuned_parameters, cv=splits,
                           scoring='%s' % scores[0],n_jobs=n_jobs)
        
                                  
        clf.fit(X, Y)
        #print("Score",clf.best_score_)
        end_rfc = time.time()
        print("Total time to process: ",end_rfc - start_rfc)
        
        with open(path_results+"parameters_rfc.txt", "a") as file:
            for item in clf.best_params_:
              file.write(" %s %s " %(item,clf.best_params_[item] ))
            file.write("\n")
        return(clf.best_params_,random_state)
        
        
        
    def TestRFC(self,X_train,Y_train,X_test,Y_test,n_estim,max_depth,max_feat,crit,m_s_split,m_s_leaf,itera,rfc_m,random_state):
        """
        This function trains and tests the RFC method on the dataset and returns training and testing AUC and the ROC values
        Input:
            X_train: training set
            Y_train: training set labels
            X_test: testing set
            Y_test: testing set
            n_estim:  number of trees
            max_feat: number of features when looking for best split
            crit: criterion for quality of split measure
            itera: iteration of cross validation, used to write down the models 
        Output: 
            result: array with training [0] and testing error[1]
            fpr_svm: false positive rate, ROc values
            tpr_svm: true positive rate, ROc values
            clf: trained classifier, can be used to combine with others, like superlearner
            probas: probabilities from predict function
            brier: brier score
        """    
         
        clf_rfc = RandomForestClassifier(max_features=max_feat,n_estimators=n_estim,max_depth=max_depth,min_samples_split=m_s_split,
                                         min_samples_leaf=m_s_leaf,oob_score = True,criterion=crit,random_state=random_state)
               
        clf_rfc.fit(X_train,Y_train)
        #print(clf.feature_importances_)
        preds = clf_rfc.predict(X_test)
        probas = clf_rfc.predict_proba(X_test)[:, 1]
        #preds=probas>0.5   
        rfc_m.clf_auc[itera]=roc_auc_score(Y_test,probas)
    
        fpr_rf, tpr_rf, _ = roc_curve(Y_test, probas)  
        rfc_m.clf_brier[itera] = brier_score_loss(Y_test, probas)   
        conf_m=confusion_matrix(Y_test, preds)
        rfc_m.clf_sens[itera]=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
        rfc_m.clf_spec[itera]=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
        rfc_m.feat_imp[itera,:]=clf_rfc.feature_importances_
        #name=('Models/RFC'+str(itera)+'.pkl')
        #joblib.dump(clf,name)
        
        return(fpr_rf,tpr_rf,probas,clf_rfc)

        
    def __init__(self,run,x_train,y_train,x_test,y_test,itera,cv,mean_tprr,rfc_m,path_results):
        if run:
            self.name='RFC'
            rfc_m.run=True
            Paramsrfc,random_state=self.RandomGridSearchRFC(x_train,y_train,cv,path_results)
            print("Done Grid Search")
            fpr_rf,tpr_rf,probas_t,self.clf=self.TestRFC(x_train,y_train,x_test,y_test,Paramsrfc['n_estimators'],Paramsrfc['max_depth'],Paramsrfc['max_features'],
                                                         Paramsrfc['criterion'],Paramsrfc['min_samples_split'],Paramsrfc['min_samples_leaf'],itera,rfc_m,random_state)
            print("Done testing - RFC", rfc_m.clf_auc[itera])
            mean_fpr = np.linspace(0, 1, 100) 
            rfc_m.mean_tpr += interp(mean_fpr, fpr_rf,tpr_rf)
            rfc_m.mean_tpr[0] = 0.0
        else:
            self.name='NONE'
            self.clf=0

class SVM_Pipeline: 

    def RandomGridSearchSVM(self,X,Y,splits,path_results):
        """
        This function looks for the best set o parameters for RFC method
        Input: 
            X: training set
            Y: labels of training set
            splits: cross validation splits, used to make sure the parameters are stable
        Output:
            clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
        """    
        
        start_svm = time.time()  
    
        tuned_parameters = {
        'C':            ([0.1, 0.01, 0.001, 1, 10, 100]),
        'kernel':       ['linear', 'rbf','poly'],                
        'degree':       ([1,2,3,4,5,6]),
        'gamma':         [1, 0.1, 0.01, 0.001, 0.0001]
        #'tol':         [1, 0.1, 0.01, 0.001, 0.0001],
        }
        
        scores = ['roc_auc']   
  
        print("SVM Grid Search")
        clf =  RandomizedSearchCV(SVC(), tuned_parameters, cv=splits,
                       scoring='%s' % scores[0],n_jobs=n_jobs)    
        
       # clf =  GridSearchCV(SVC(), tuned_parameters, cv=splits,
       #                scoring='%s' % scores[0],n_jobs=-1)    
        clf.fit(X, Y)
    
        end_svm = time.time()
        print("Total time to process: ",end_svm - start_svm)
        #print("Score",clf.best_score_)
        with open(path_results+"parameters_svm.txt", "a") as file:
            for item in clf.best_params_:
              file.write(" %s %s " %(item,clf.best_params_[item] ))
            file.write("\n")
        return(clf.best_params_)
        
        
        
    def TestSVM(self,X_train,Y_train,X_test,Y_test,kernel,C,gamma,deg,itera,svm_m):
        """
        This function trains and tests the SVM method on the dataset and returns training and testing AUC and the ROC values
        Input:
            X_train: training set
            Y_train: training set labels
            X_test: testing set
            Y_test: testing set
            n_estim:  number of trees
            max_feat: number of features when looking for best split
            crit: criterion for quality of split measure
            itera: iteration of cross validation, used to write down the models 
        Output: 
            result: array with training [0] and testing error[1]
            fpr_svm: false positive rate, ROc values
            tpr_svm: true positive rate, ROc values
            clf: trained classifier, can be used to combine with others, like superlearner
            probas: probabilities from predict function
            brier: brier score
        """    

         
        clf_svm = svm.SVC(C=C,kernel=kernel,gamma=gamma,degree=deg,probability=True)
               
        clf_svm.fit(X_train,Y_train)
        preds = clf_svm.predict(X_test)
        decisions = clf_svm.decision_function(X_test)
        probas=\
        (decisions-decisions.min())/(decisions.max()-decisions.min())
        preds=probas>0.5
        #probas=clf.predict_proba(X_test)[:, 1]
        svm_m.clf_auc[itera]=roc_auc_score(Y_test,probas)
    
        fpr_rf, tpr_rf, _ = roc_curve(Y_test, probas)  
        svm_m.clf_brier[itera] = brier_score_loss(Y_test, probas)   
        conf_m=confusion_matrix(Y_test, preds)
        svm_m.clf_sens[itera]=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
        svm_m.clf_spec[itera]=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
        
        #name=('Models/RFC'+str(itera)+'.pkl')
        #joblib.dump(clf,name)
        
    
        return(fpr_rf,tpr_rf,probas,clf_svm)

        
    def __init__(self,run,x_train,y_train,x_test,y_test,itera,svm_m,cv,mean_tprr,path_results):
        if run:
            self.name='SVM'
            svm_m.run=True
            Paramssvm=self.RandomGridSearchSVM(x_train,y_train,cv,path_results)
            fpr_rf,tpr_rf,probas_t,self.clf=self.TestSVM(x_train,y_train,x_test,y_test,Paramssvm.get('kernel'),Paramssvm.get('C'),Paramssvm.get('gamma'),Paramssvm.get('degree'),itera,svm_m)
            print("Done testing - SVM", svm_m.clf_auc[itera])
            mean_fpr = np.linspace(0, 1, 100) 
            svm_m.mean_tpr += interp(mean_fpr, fpr_rf,tpr_rf)
            svm_m.mean_tpr[0] = 0.0
        else:
            self.name='NONE'
            self.clf=0

class NN_Pipeline: 

    def RandomGridSearchNN(self,X,Y,splits,path_results):
        """
        This function looks for the best set o parameters for RFC method
        Input: 
            X: training set
            Y: labels of training set
            splits: cross validation splits, used to make sure the parameters are stable
        Output:
            clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
        """    
        
        start_nn = time.time()  
    
          
        scores = ['roc_auc']
        start_nn = time.time()  
        tuned_parameters = {
        'activation': (['relu','logistic']),
        'hidden_layer_sizes':([[80,160,80],[78,156,78],[88,176,88],[80,160]]),
        #'hidden_layer_sizes':([[131,191,131],[131,231,131],[131,131,131]]),
        'alpha':     ([0.01, 0.001, 0.0001]),
        'batch_size':         [32,64],
        'learning_rate_init':    [0.01, 0.001],
        'solver': ["adam"]}
        
        scores = ['roc_auc']   
           
        print("NN Grid Search")
        mlp = MLPClassifier(max_iter=5000) 
        clf = RandomizedSearchCV(mlp, tuned_parameters, cv= splits, scoring='%s' % scores[0],n_jobs=n_jobs+1)
            
        clf.fit(X, Y)
             
        end_nn = time.time()
        print("Total time to process NN: ",end_nn - start_nn)
        with open(path_results+"parameters_NN.txt", "a") as file:
            for item in clf.best_params_:
              file.write(" %s %s " %(item,clf.best_params_[item] ))
            file.write("\n")
        return(clf.best_params_)
        
        
        
                
    def TestNN(self,X_train,Y_train,X_test,Y_test,act,hid,alpha,batch,learn,solver,itera,nn_m):
        """
        This function trains and tests the SVM method on the dataset and returns training and testing AUC and the ROC values
        Input:
            X_train: training set
            Y_train: training set labels
            X_test: testing set
            Y_test: testing set
            act:  activation function for the hidden layers
            hid: size of hidden layers, array like [10,10]
            alpha: regularization parameters
            batch: minibatch size
            learn: learning rate
            solver: Adam or SGD
            itera: iteration of cross validation, used to write down the models 
        Output: 
            result: array with training [0] and testing error[1]
            fpr_svm: false positive rate, ROc values
            tpr_svm: true positive rate, ROc values
            clf: trained classifier, can be used to combine with others, like superlearner
            probas: probabilities from predict function
            brier: brier score
        """

   
          
        clf_nn=MLPClassifier(solver=solver,activation=act,hidden_layer_sizes=hid,alpha=alpha,
                             batch_size=batch,learning_rate_init=learn,max_iter=5000)
                    
        clf_nn = clf_nn.fit(X_train, Y_train)
        preds = clf_nn.predict(X_test)
        probas = clf_nn.predict_proba(X_test)[:, 1]
                  
        nn_m.clf_auc[itera]=roc_auc_score(Y_test,probas)
    
        fpr_rf, tpr_rf, _ = roc_curve(Y_test, probas)  
        nn_m.clf_brier[itera] = brier_score_loss(Y_test, probas)   
        conf_m=confusion_matrix(Y_test, preds)
        nn_m.clf_sens[itera]=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
        nn_m.clf_spec[itera]=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
        
        #name=('Models/RFC'+str(itera)+'.pkl')
        #joblib.dump(clf,name)
        
    
        return(fpr_rf,tpr_rf,probas,clf_nn)

        
    def __init__(self,run,x_train,y_train,x_test,y_test,itera,nn_m,cv,mean_tprr,path_results):
        if run:
            self.name='NN'
            nn_m.run=True
            Paramsnn=self.RandomGridSearchNN(x_train,y_train,cv,path_results)
            fpr_rf,tpr_rf,probas_t,self.clf=self.TestNN(x_train,y_train,x_test,y_test,Paramsnn.get('activation'),Paramsnn.get('hidden_layer_sizes'),Paramsnn.get('alpha'),Paramsnn.get('batch_size'),
                                               Paramsnn.get('learning_rate_init'),Paramsnn.get('solver'),itera,nn_m)
            print("Done testing - NN", nn_m.clf_auc[itera])
            mean_fpr = np.linspace(0, 1, 100) 
            nn_m.mean_tpr += interp(mean_fpr, fpr_rf,tpr_rf)
            nn_m.mean_tpr[0] = 0.0
        else:
            self.name='NONE'
            self.clf=0
                

class LR_Pipeline: 
    
    def Feature_Selection(self,X,y,T,method,cv):
        """
        This functions returns only the features selected by the method using the threshold selected.
        We advise to run this function with several thresholds and look for the best, put this function inside a loop and see how it goes
        Suggestions for the range of t, thresholds = np.linspace(0.00001, 0.1, num=10)
        Input: 
            X=training set
            y=training labels
            T=threshold selected
            which method= 'rfc', 'lasso', 'elastik'
            cv= number of cross validation iterations
        Output:
        Boolean array with the selected features,with this you can X=X[feats] to select only the relevant features
        """
        alphagrid = np.linspace(0.001, 0.99, num=cv)
        
        clf = {
              'rfc': RandomForestClassifier(),
              'lasso': LassoCV(),#alphas=alphagrid),
              'elastik': ElasticNetCV(alphas=alphagrid),
              'backward': RFECV(LogisticRegression(),cv=cv,n_jobs=n_jobs)
                
        }[method]
        if method=='backward':
            clf = clf.fit(X, y)
            feats=clf.support_
        else:
            clf.fit(X,y)
            sfm = SelectFromModel(clf)#, threshold=T)
            print(X.shape)
            sfm.fit(X,y)                                            
            feats=sfm.get_support()

        return(feats)

    
    def TestLogistic(self,X_train,Y_train,X_test,Y_test,itera,lr_m,feats):
                
        clf = LogisticRegression(C=100000,solver="liblinear")
             
        clf.fit(X_train,Y_train)
    
        preds = clf.predict(X_test)
        probas = clf.predict_proba(X_test)[:, 1]
        preds=probas>0.5   
        lr_m.clf_auc[itera]=roc_auc_score(Y_test,probas)
    
        fpr_rf, tpr_rf, _ = roc_curve(Y_test, probas)  
        lr_m.clf_brier[itera] = brier_score_loss(Y_test, probas)   
        conf_m=confusion_matrix(Y_test, preds)
        lr_m.clf_sens[itera]=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
        lr_m.clf_spec[itera]=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
        
        odds=np.exp(clf.coef_)
        feats=np.array(feats,dtype='float64')
        pos=0
        for i in range(0,feats.shape[0]):
            if feats[i]==1:
                feats[i]=odds[0,pos]
                #print(odds[0,pos])
                pos=pos+1
        #print(feats)
        lr_m.feat_imp[itera,:]=feats       
        print("classes",clf.classes_)        
        #name=('Models/RFC'+str(itera)+'.pkl')
        #joblib.dump(clf,name)
        
        return(fpr_rf,tpr_rf,probas,clf)
        
    def __init__(self,run,x_train,y_train,x_test,y_test,itera,mean_tprr,lr_m,cv,select_feats_logit=False,T=0.1,method='rfc'):
        feats=np.ones(x_train.shape[1])
        if run:
            lr_m.run=True
            self.name='LR'
            if select_feats_logit:
                feats=self.Feature_Selection(x_train,y_train,T,method,cv)
                print("Features Selected",sum(feats))
                x_train=x_train[:,feats]
                x_test=x_test[:,feats]
    

            fpr_lr,tpr_lr,probas_t,self.clf=self.TestLogistic(x_train,y_train,x_test,y_test,itera,lr_m,feats)
            print("Done testing - LR", lr_m.clf_auc[itera])
            mean_fpr = np.linspace(0, 1, 100) 
            lr_m.mean_tpr += interp(mean_fpr, fpr_lr,tpr_lr)
            lr_m.mean_tpr[0] = 0.0
        else:
            self.name='NONE'
            self.clf=0

class SL_Pipeline:
    
    def TrainAndTestSL(self,X_train,Y_train,X_test,Y_test,itera,sl_m,class_list,libnames):

            sl_clf=sl.SuperLearner(class_list, libnames, loss="nloglik")
            #sl_clf=sl.SuperLearner(lib, libnames, loss="L2")
            
            sl_clf.fit(X_train,Y_train)
    
            probas = sl_clf.predict(X_test)
            preds=probas>0.5   
            sl_m.clf_auc[itera]=roc_auc_score(Y_test,probas)
        
            fpr_rf, tpr_rf, _ = roc_curve(Y_test, probas)  
            sl_m.clf_brier[itera] = brier_score_loss(Y_test, probas)   
            conf_m=confusion_matrix(Y_test, preds)
            sl_m.clf_sens[itera]=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
            sl_m.clf_spec[itera]=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
            return(fpr_rf,tpr_rf,probas)
            
    def __init__(self,run,x_train,y_train,x_test,y_test,l,sl_m,mean_tprr,class_rfc,class_svm,class_lr,class_nn):   
        if run:
            list_clfs=[class_rfc,class_svm,class_lr,class_nn]
            list_clfs=[x.clf for x in list_clfs if x.clf != 0]
            names=[class_rfc.name,class_svm.name,class_lr.name,class_nn.name]
            names=[x for x in names if x != 'NONE'] 
            self.name='SL'
            sl_m.run=True
            fpr_sl,tpr_sl,probas_t=self.TrainAndTestSL(x_train,y_train,x_test,y_test,l,sl_m,list_clfs,names)
            print("Done testing - SL", sl_m.clf_auc[l])
            mean_fpr = np.linspace(0, 1, 100) 
            sl_m.mean_tpr += interp(mean_fpr, fpr_sl,tpr_sl)
            sl_m.mean_tpr[0] = 0.0
        else:
            self.name='NONE'
          
            


def Mean_Confidence_Interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def Print_Results(m,splits,names,path_results):    
    colors=['darkorange','blue','green','black','yellow']
    path_results_txt=path_results+"Results.txt"
    for i in range(0,len(names)):        
        with open(path_results_txt, "a") as file:
              file.write("Results %s \n" %(names[i])) 
              file.write("Average AUC %0.4f CI  %0.4f - %0.4f \n" %(Mean_Confidence_Interval(m[i].clf_auc)))              
              file.write("Average Sensitivity %0.4f CI  %0.4f - %0.4f \n" %(Mean_Confidence_Interval(m[i].clf_sens)))
              file.write("Average Specificity %0.4f CI  %0.4f - %0.4f \n" %(Mean_Confidence_Interval(m[i].clf_spec)))
              file.write("\n")
        np.save(file=path_results+'AUCs_'+names[i]+'.npy',arr=m[i].clf_auc)
        mean_tpr=m[i].mean_tpr
        mean_tpr /= splits
        mean_tpr[-1] = 1.0
        #frac_pos_rfc  /= skf.get_n_splits(X, Y)
        mean_fpr = np.linspace(0, 1, 100) 
        mean_auc_rfc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color=colors[i],lw=2, label=names[i]+' (area = %0.2f)' % mean_auc_rfc)
        plt.legend(loc="lower right")
        np.save(file=path_results+'tpr_'+names[i]+'.npy',arr=mean_tpr)
        np.save(file=path_results+'fpr_'+names[i]+'.npy',arr=mean_fpr)
        if names[i]=='RFC' or names[i]=='LR':
            np.save(file=path_results+'Feat_Importance'+names[i]+'.npy',arr=m[i].feat_imp)
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
    plt.show()  
  
    
def Print_Results_Excel(m,splits,names,path_results):    
    colors=['darkorange','blue','green','black','yellow']
    book = xlwt.Workbook(encoding="utf-8")    
    sheet1 = book.add_sheet("Sheet 1")
    path_results_txt=path_results+"Results.xls"

    sheet1.write(0, 0, "Methods")
    sheet1.write(0, 1, "AUC 95% CI ")
    sheet1.write(0, 2, "Sensitivity")
    sheet1.write(0, 3, "Specificity")
    #Spec and sensitivty are inverted because of the label
    for i in range(0,len(names)):        
    
        sheet1.write(i+1,0,(names[i])) 
        sheet1.write(i+1,1,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_auc))))              
        sheet1.write(i+1,2,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_sens))))              
        sheet1.write(i+1,3,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_spec))))              

        np.save(file=path_results+'AUCs_'+names[i]+'.npy',arr=m[i].clf_auc)
        mean_tpr=m[i].mean_tpr
        mean_tpr /= splits
        mean_tpr[-1] = 1.0
        #frac_pos_rfc  /= skf.get_n_splits(X, Y)
        mean_fpr = np.linspace(0, 1, 100) 
        mean_auc_rfc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color=colors[i],lw=2, label=names[i]+' (area = %0.2f)' % mean_auc_rfc)
        plt.legend(loc="lower right")
        np.save(file=path_results+'tpr_'+names[i]+'.npy',arr=mean_tpr)
        np.save(file=path_results+'fpr_'+names[i]+'.npy',arr=mean_fpr)
    book.save(path_results_txt)        
    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
    #plt.show() 

