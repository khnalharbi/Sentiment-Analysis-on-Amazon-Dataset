
import numpy as np
import pandas as pd
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

class TrAdaBoost(object):
    def __init__(self,N=10,base_estimator=DecisionTreeClassifier(),score=roc_auc_score):    
        self.N=N
        self.base_estimator=base_estimator
        self.score=score
        self.beta_all = None
        self.estimators=[]
            
    def _calculate_weights(self,weights): 
        weights = weights.ravel()     
        total = np.sum(weights)   
        print("Total weight is: ",total," min Weight is: ",np.min(weights)," max Weight is: ",np.max(weights))   
        return np.asarray(weights / total, order='C')      
                    
    def _calculate_error_rate(self,y_true, y_pred, weight):      
        weight = weight.ravel()
        total = np.sum(weight) 
        print("Total weight is: ",total," min Weight is: ",np.min(weight)," max Weight is: ",np.max(weight))     
        return np.sum(weight / total * np.abs(y_true - y_pred))      
             
    def fit(self,source,target,source_label,target_label):
        source_shape=source.shape[0]
        target_shape=target.shape[0]
        trans_data = np.concatenate((source, target), axis=0)      
        trans_label = np.concatenate((source_label,target_label), axis=0)      
        weights_source = np.ones([source_shape, 1])/source_shape      
        weights_target = np.ones([target_shape, 1])/target_shape
        weights = np.concatenate((weights_source, weights_target), axis=0)
        
        bata = 1 / (1 + np.sqrt(2 * np.log(source_shape / self.N)))    
        self.beta_all = np.zeros([1, self.N])
        result_label = np.ones([source_shape+target_shape, self.N])    

        trans_data = np.asarray(trans_data, order='C')
        trans_label = np.asarray(trans_label, order='C')     
        
        best_round = 0
        score=0
        flag=0
        
        for i in tqdm(range(self.N)):      
            P = self._calculate_weights(weights) 
            est = clone(self.base_estimator).fit(trans_data,trans_label,sample_weight=P.ravel())
            self.estimators.append(est)
            y_preds=est.predict(trans_data)
            result_label[:, i]=y_preds

            y_target_pred=est.predict(target)
            error_rate = self._calculate_error_rate(target_label, y_target_pred,  \
                                              weights[source_shape:source_shape + target_shape, :])  
                  
            if error_rate >= 0.5 or error_rate == 0:      
                self.N = i
                print('early stop! due to error_rate=%.2f'%(error_rate))      
                break       

            self.beta_all[0, i] = error_rate / (1 - error_rate)      
     
            for j in range(target_shape):      
                weights[source_shape + j] = weights[source_shape + j] * \
                np.power(self.beta_all[0, i],(-np.abs(result_label[source_shape + j, i] - target_label[j])))
  
            for j in range(source_shape):      
                weights[j] = weights[j] * np.power(bata,np.abs(result_label[j, i] - source_label[j]))
                
            tp=self.score(target_label,y_target_pred)
            print('The '+str(i)+' rounds score is '+str(tp))

    def _predict_one(self, x):
        """
        Output the hypothesis for a single instance
        :param x: array-like
            target label of a single instance from each iteration in order
        :return: 0 or 1
        """
        x, N = check_array(x, ensure_2d=False), self.N
        # replace 0 by 1 to avoid zero division and remove it from the product
        beta = [self.beta_all[0,t] if self.beta_all[0,t] != 0 else 1 for t in range(int(np.ceil(N/2)), N)]
        cond = np.prod([b ** -x for b in beta]) >= np.prod([b ** -0.5 for b in beta])
        return int(cond)

    def predict(self, x_test):
        y_pred_list = np.array([est.predict(x_test) for est in self.estimators]).T
        y_pred = np.array(list(map(self._predict_one, y_pred_list)))
        return y_pred