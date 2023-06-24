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

###################### Supervised Learning ######################
# Fit the model on training set
clf_SVM = LinearSVC(penalty = 'l2')
clf_SVM.fit(X_train_h, Y_train_h)
# save the model to disk
filename = 'clf_SVM.sav'
pickle.dump(clf_SVM, open(filename, 'wb'))

###################### Semi-supervised Learning ######################
# data for Semi- supervised Learning
x_h_l = X_train_h_r[:8,:]
y_h_l = Y_train_h_r[:8]
y_h_labeled = np.where(y_h_l == 0 , -1, 1)

x_h_u = X_train_h_r[8:,:]
y_h_u = Y_train_h_r[8:]
y_h_un = np.where(y_h_u  == 0 , -1, 1)
y_h_test = np.where(Y_test_h_r  == 0 , -1, 1)

# Fit the model on training set
x_uninon_train = np.concatenate((x_h_l,x_h_u), axis=0)
y_uninon_train = np.concatenate((y_h_l, y_h_u), axis=0)

clf_SSL = LabelSpreading()
clf_SSL.fit(x_uninon_train, y_uninon_train)

# save the model to disk
filename = 'clf_SSL.sav'
pickle.dump(clf_SSL, open(filename, 'wb'))

###################### Unsupervised Learning ######################
# Fit the model on training set
centroid, label = kmeans2(X_train_h_r, k = 2, seed=22)

# save the model to disk
filename = 'centroid.sav'
pickle.dump(centroid, open(filename, 'wb'))

filename = 'label.sav'
pickle.dump(label, open(filename, 'wb'))
###################### Transfer Learning ######################

X_train_h_r, X_test_h_r, Y_train_h_r, Y_test_h_r = train_test_split(X_review_h, y_h, train_size = 3000, test_size=600,random_state=200, stratify =y_h)

# Fit the model on training set
base_estimator = LinearSVC(penalty = 'l2')
clf_TR = TrAdaBoost(N=4,base_estimator=base_estimator,score=accuracy_score)
clf_TR.fit(X_train_h_r, X_train_k_r, Y_train_h_r, Y_train_k_r)
# save the model to disk
filename = 'clf_TR.sav'
pickle.dump(clf_TR, open(filename, 'wb'))
