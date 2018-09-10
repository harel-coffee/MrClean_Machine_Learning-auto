# -*- coding: utf-8 -*-
"""
This code compared 100 iterations from my results, as Koos suggested
"""
import scipy.stats as sp
import numpy as np


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

path=r'C:\Users\laramos\Dropbox\Mrcleanlearning\results\New100-All_contscore-mrs-rfc\\'
path_logit=r"C:\Users\laramos\Dropbox\Mrcleanlearning\results\MRS\100-Knowledge_all-mrs-rfc\\"
#path_logit="E:\\Results MRs\\Results postici\\LR-Center-Baseline_contscore-posttici_c-lasso\\"

#auc_svm=np.loadtxt('E:\\PhdRepos\\ClinicalNeuralNetworkSVM\\Prospective\\Koos suggestion and final results\\ML\\ROC-SVM.txt')
auc_rfc=np.load(path+"AUCs_RFC.npy")
#auc_logit=np.loadtxt('E:\\PhdRepos\\ClinicalNeuralNetworkSVM\\Prospective\\Koos suggestion and final results\\ML\\ROC-Logit.txt')
#auc_nn=np.loadtxt('E:\\PhdRepos\\ClinicalNeuralNetworkSVM\\Prospective\\Koos suggestion and final results\\ML\\ROC-NN.txt')
#auc_sl=np.loadtxt(path+"AUC-Logit.txt")


auc_logit=np.load(path_logit+"AUCs_LR.npy")





dif_rfc_logit=auc_rfc-auc_logit
#dif_sl_logit=auc_sl-auc_logit

mean_rfc,conf1_rfc,conf2_rfc=mean_confidence_interval(dif_rfc_logit,confidence=0.95)
#mean_sl,conf1_sl,conf2_sl=mean_confidence_interval(dif_sl_logit,confidence=0.975)



print('RFC Average= {0:.3f} Confidence Interval {1:.3f} to {2:.3f} '.format(mean_rfc,conf1_rfc,conf2_rfc))
#print('SL Average= {0:.3f} Confidence Interval {1:.3f} - {2:.3f} '.format(mean_sl,conf1_sl,conf2_sl))

