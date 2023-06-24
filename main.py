import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
import random
import warnings
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

from tqdm import tqdm
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import clone
from sklearn.utils.validation import check_array
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelSpreading
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.spatial import distance
from scipy.cluster.vq import kmeans2
from scipy.stats import multivariate_normal
from TrAdaboost import TrAdaBoost

# read Data
X_review_h = np.load('Data/X_home_review_wv.npy', allow_pickle = True)
y_h = np.load('Data/y_data_home_finalized.npy', allow_pickle = True)

X_review_k = np.load('Data/X_Kitch_review_wv.npy', allow_pickle = True)
y_k = np.load('Data/y_data_Kitch_finalized.npy' ,allow_pickle = True)

# split train and test
# Train split for Supervised Models
X_train_h, X_test_h, Y_train_h, Y_test_h = train_test_split(X_review_h, y_h, test_size=0.2, random_state=200) # home
X_train_k, X_test_k, Y_train_k, Y_test_k = train_test_split(X_review_k, y_k, test_size=0.2, random_state=200) # kitchen
# Reduced data for TL & SSL & USL
X_train_h_r, X_test_h_r, Y_train_h_r, Y_test_h_r = train_test_split(X_review_h, y_h, train_size = 1000, test_size=300,random_state=200, stratify =y_h)
X_train_k_r, X_test_k_r, Y_train_k_r, Y_test_k_r = train_test_split(X_review_k, y_k, train_size = 400, test_size=200,random_state=200, stratify =y_k)

################################# Supervised Learning #################################
print('\n###################### Supervised Learning ######################')
filename = 'clf_SVM.sav'
clf_SVM = pickle.load(open(filename, 'rb'))

print("SVM model report on training and testing data: \n")
training_report = classification_report(Y_train_h, clf_SVM.predict(X_train_h), output_dict=False)
testing_report = classification_report(Y_test_h, clf_SVM.predict(X_test_h), output_dict=False)
print("The training report is:\n", training_report)
print("The testing report is:\n", testing_report)

################################# Semi-supervised Learning #################################
print('\n###################### Semi-supervised Learning ######################')

filename = 'clf_SSL.sav'
clf_SSL = pickle.load(open(filename, 'rb'))

acc_score = clf_SSL.score(X_test_h_r, Y_test_h_r)
print(f'The accuracy score of semi_supervised by LabelSpreading on test data home is= {acc_score:.3f}')


#################################Unsupervised Learning #################################
print('\n###################### Unsupervised Learning ######################')
filename = 'centroid.sav'
centroid = pickle.load(open(filename, 'rb'))
filename = 'label.sav'
label = pickle.load(open(filename, 'rb'))

y_pred1 = np.zeros((len(X_test_h_r)))

for i in range(len(X_test_h_r)):
    if distance.euclidean( X_test_h_r[i,:] , centroid[0,:] ) < distance.euclidean( X_test_h_r[i,:] , centroid[1,:] ):
        y_pred1[i] = 0
        #y_e_r = y[label == i]
    elif distance.euclidean( X_test_h_r[i,:] , centroid[0,:] ) >= distance.euclidean( X_test_h_r[i,:] ,centroid[1,:]):
        y_pred1[i] = 1
    else:
        y_pred1[i] = None
   
acc_scc = accuracy_score(Y_train_h_r, label)
print(f'The accuracy of kmeans for USL on train data home is = {acc_scc:.3f}')

acc_scc = accuracy_score(Y_test_h_r, y_pred1)
print(f'The accuracy of kmeans for USL on test data home is = {acc_scc:.3f}')

################################# Transfer Learning #################################
print('\n###################### Transfer Learning ######################')

X_train_h_r, X_test_h_r, Y_train_h_r, Y_test_h_r = train_test_split(X_review_h, y_h, train_size = 3000, test_size=600,random_state=200, stratify =y_h)

filename = 'clf_TR.sav'
clf_TR = pickle.load(open(filename, 'rb'))

yt_pred = clf_TR.predict(X_train_k_r)
print('The accuracy of TL in target domain for train data is:',accuracy_score(Y_train_k_r, yt_pred))

yt_test_pred = clf_TR.predict(X_test_k_r)
print('The testing report of TL in target domain is:\n',classification_report(Y_test_k_r, yt_test_pred, output_dict=False))
