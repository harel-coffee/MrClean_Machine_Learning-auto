# MrClean_Machine_Learning
Pipeline for predicting ischemic stroke functional outcome and e-tici using the MrClean registry dataset

This folder includes the python code for the analysis of the MrClean dataset.

Data_Preprocessing.py: includes a list of functions for pre-processing the dataset, cleaning and imputing missing data, it also works for multiple files extensions.
Feature_Selection: includes many feature selection techniques.
Methods.py: includes all the classifiers classes used in this study.
Pipeline: Includes mutiple calss to the methods, controls the iterations and cross-validation, this is the main file.
Results_comp.py: For comparing the final results
combine_ROC: for plotting nice ROC figures from all classifiers.

If you use our code please cite: https://www.frontiersin.org/articles/10.3389/fneur.2018.00784/abstract
