import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (roc_auc_score,
                             accuracy_score,
                             recall_score,
                             confusion_matrix)
from .smoothing import smoothing

def logistic_regression(t, test_x) :
    lr = LogisticRegression(random_state = 0, max_iter=500, solver='newton-cg') 
    lr.fit(t[0].reset_index(drop=True), t[1].reset_index(drop=True)) # training 
    lr_pred = lr.predict_proba(test_x) # test 셋을 학습한 모델에 넣어서 확률값을 받는다. 
    lr_sm_pred = smoothing(lr_pred[:,1]).reset_index(drop=True)  # 확률값들을 스무딩한다. 
    lr_origin = pd.DataFrame(lr_pred[:,1]).reset_index(drop=True)  
    return lr_sm_pred, lr_origin

def random_forest(t, test_x) :
    rf = RandomForestClassifier(max_depth = 50 ,random_state=0)
    rf.fit(t[0].reset_index(drop=True), t[1].reset_index(drop=True))
    rf_pred = rf.predict_proba(test_x)
    rf_sm_pred = smoothing(rf_pred[:,1])  # 확률값들을 스무딩한다. 
    rf_origin = pd.DataFrame(rf_pred[:,1]).reset_index(drop=True)
    return rf_sm_pred, rf_origin

def neural_network(t, true_x) :
    nn = MLPClassifier(hidden_layer_sizes=3 , max_iter=100)
    nn.fit(t[0].reset_index(drop=True), t[1].reset_index(drop=True))
    nn_pred = nn.predict_proba(true_x)
    nn_sm_pred = smoothing(nn_pred[:,1])  # 확률값들을 스무딩한다. 
    nn_origin = pd.DataFrame(nn_pred[:,1]).reset_index(drop=True)
    return nn_sm_pred, nn_origin

def gradient_boosting(t, true_x) :
    grb = GradientBoostingClassifier(n_estimators=50, random_state=0)
    grb.fit(t[0].reset_index(drop=True), t[1].reset_index(drop=True))
    grb_pred = grb.predict_proba(true_x)
    grb_sm_pred = smoothing(grb_pred[:,1])  # 확률값들을 스무딩한다. 
    grb_origin = pd.DataFrame(grb_pred[:,1]).reset_index(drop=True)    
    return grb_sm_pred, grb_origin

def LDA(t, true_x) :
    lda = LinearDiscriminantAnalysis()
    lda.fit(t[0].reset_index(drop=True), t[1].reset_index(drop=True))
    lda_pred = lda.predict_proba(true_x)
    lda_sm_pred = smoothing(lda_pred[:,1])
    lda_origin = pd.DataFrame(lda_pred[:,1]).reset_index(drop=True)
    return lda_sm_pred, lda_origin


# AUC function definition
def AUC(test, pred):
    auc = roc_auc_score(test, pred) 
    return auc

# Accuracy function definition
def Accuracy(test, pred):
    accur = accuracy_score(test, pred)
    return accur

# Confusion function definition 
def Sensitivity(test, pred):
    conf_mat = confusion_matrix(test, pred)
    sen = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0])
    return sen

def Specificity(test, pred):
    conf_mat = confusion_matrix(test, pred)
    spec = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1])   
    return spec

def result(test, pred):
    m = AUC(test, pred)
    n = Accuracy(test, pred)
    l = Sensitivity(test, pred)
    p = Specificity(test, pred) 
    rslt = [m, n, l, p]
    df_result = pd.DataFrame(rslt, index = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity'])
    df_result.columns = ['result']
    return df_result