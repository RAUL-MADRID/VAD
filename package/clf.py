import pandas as pd
import pickle

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier,
                              VotingClassifier,
                              StackingClassifier)

from sklearn.metrics import (roc_auc_score,
                             accuracy_score,
                             recall_score,
                             confusion_matrix)
from .smoothing import smoothing

def logistic_regression(t, test_x, mode='train') :
    filename = './models/lr_clf_vad.pkl'
    if mode == 'train':
        lr = LogisticRegression(random_state = 0, max_iter=500, solver='newton-cg') 
        lr.fit(t[0].reset_index(drop=True), t[1].reset_index(drop=True)) # training 
        pickle.dump(lr, open(filename, 'wb'))
    else:
        lr =  pickle.load(open(filename, 'rb'))
    lr_pred = lr.predict_proba(test_x) # test 셋을 학습한 모델에 넣어서 확률값을 받는다. 
    lr_sm_pred = smoothing(lr_pred[:,1]).reset_index(drop=True)  # 확률값들을 스무딩한다. 
    lr_origin = pd.DataFrame(lr_pred[:,1]).reset_index(drop=True)  
    return lr_sm_pred, lr_origin

def random_forest(t, test_x, mode='train') :
    filename = './models/rf_clf_vad.pkl'
    if mode == 'train':
        rf = RandomForestClassifier(max_depth = 50 ,random_state=0)
        rf.fit(t[0].reset_index(drop=True), t[1].reset_index(drop=True))
        pickle.dump(rf, open(filename, 'wb'))
    else:
        rf = pickle.load(open(filename, 'rb'))
    rf_pred = rf.predict_proba(test_x)
    rf_sm_pred = smoothing(rf_pred[:,1])  # 확률값들을 스무딩한다. 
    rf_origin = pd.DataFrame(rf_pred[:,1]).reset_index(drop=True)
    return rf_sm_pred, rf_origin

def neural_network(t, true_x, mode='train') :
    filename = './models/mlp_clf_vad.pkl'
    if mode == 'train':
        nn = MLPClassifier(hidden_layer_sizes=3 , max_iter=100)
        nn.fit(t[0].reset_index(drop=True), t[1].reset_index(drop=True))
        pickle.dump(nn, open(filename, 'wb'))
    else:
        nn = pickle.load(open(filename, 'rb'))
    nn_pred = nn.predict_proba(true_x)
    nn_sm_pred = smoothing(nn_pred[:,1])  # 확률값들을 스무딩한다. 
    nn_origin = pd.DataFrame(nn_pred[:,1]).reset_index(drop=True)
    return nn_sm_pred, nn_origin

def gradient_boosting(t, true_x, mode='train') :
    filename = './models/grb_clf_vad.pkl'
    if mode == 'train':
        grb = GradientBoostingClassifier(n_estimators=50, random_state=0)
        grb.fit(t[0].reset_index(drop=True), t[1].reset_index(drop=True))
        pickle.dump(grb, open(filename, 'wb'))
    else:
        grb = pickle.load(open(filename, 'rb'))
    grb_pred = grb.predict_proba(true_x)
    grb_sm_pred = smoothing(grb_pred[:,1])  # 확률값들을 스무딩한다. 
    grb_origin = pd.DataFrame(grb_pred[:,1]).reset_index(drop=True)    
    return grb_sm_pred, grb_origin

def LDA(t, true_x, mode='train') :
    filename = './models/lda_clf_vad.pkl'
    if mode == 'train':
        lda = LinearDiscriminantAnalysis()
        lda.fit(t[0].reset_index(drop=True), t[1].reset_index(drop=True))
        pickle.dump(lda, open(filename, 'wb'))
    else:
        lda = pickle.load(open(filename, 'rb'))
    lda_pred = lda.predict_proba(true_x)
    lda_sm_pred = smoothing(lda_pred[:,1])
    lda_origin = pd.DataFrame(lda_pred[:,1]).reset_index(drop=True)
    return lda_sm_pred, lda_origin


# def lightGBM(t, true_x, mode='train') :
#     filename = './models/lgb_clf_vad.pkl'
#     if mode == 'train':
#         lgb_clf = LGBMClassifier()
#         lgb_clf.fit(t[0].reset_index(drop=True), t[1].reset_index(drop=True))
#         pickle.dump(lgb_clf, open(filename, 'wb'))
#     else:
#         lgb_clf = pickle.load(open(filename, 'rb'))
#     lgb_pred = lgb_clf.predict_proba(true_x)
#     lgb_sm_pred = smoothing(lgb_pred[:,1])
#     lgb_origin = pd.DataFrame(lgb_pred[:,1]).reset_index(drop=True)
#     return lgb_sm_pred, lgb_origin


def voting_classifier(t, true_x, mode='train'):
    filename = './models/voting_clf_vad.pkl'
    if mode == 'train':
        lr = './models/lr_clf_vad.pkl'
        #lgb = './models/lgb_clf_vad.pkl'
        nn = './models/mlp_clf_vad.pkl'
        grb = './models/grb_clf_vad.pkl'
        lr = pickle.load(open(lr, 'rb'))
        #lgb = pickle.load(open(lgb, 'rb'))
        nn = pickle.load(open(nn, 'rb'))
        grb = pickle.load(open(grb, 'rb'))
        voting_clf = VotingClassifier(
                        estimators = [('lr', lr),  
                                      ('nn', nn), 
                                      ('grb', grb)],
                        voting = 'soft')
        voting_clf.fit(t[0].reset_index(drop=True), t[1].reset_index(drop=True))
        pickle.dump(voting_clf, open(filename, 'wb'))
    else:
        voting_clf = pickle.load(open(filename, 'rb'))
    voting_pred = voting_clf.predict_proba(true_x)
    voting_sm_pred = smoothing(voting_pred[:,1])
    voting_origin = pd.DataFrame(voting_pred[:,1]).reset_index(drop=True)
    return voting_sm_pred, voting_origin


def stacking_classifier(t, true_x, mode='train'):
    filename = './models/stacking_clf_vad.pkl'
    if mode == 'train':
        lr = './models/lr_clf_vad.pkl'
        #lgb = './models/lgb_clf_vad.pkl'
        nn = './models/mlp_clf_vad.pkl'
        grb = './models/grb_clf_vad.pkl'
        lr = pickle.load(open(lr, 'rb'))
        #lgb = pickle.load(open(lgb, 'rb'))
        nn = pickle.load(open(nn, 'rb'))
        grb = pickle.load(open(grb, 'rb'))
        stacking_clf = StackingClassifier(
                        estimators = [('lr', lr), 
                                      ('nn', nn), 
                                      ('grb', grb)],
                        final_estimator=LogisticRegression(random_state = 0, max_iter=500, solver='newton-cg'))
        stacking_clf.fit(t[0].reset_index(drop=True), t[1].reset_index(drop=True))
        pickle.dump(stacking_clf, open(filename, 'wb'))
    else:
        stacking_clf = pickle.load(open(filename, 'rb'))
    stacking_pred = stacking_clf.predict_proba(true_x)
    stacking_sm_pred = smoothing(stacking_pred[:,1])
    stacking_origin = pd.DataFrame(stacking_pred[:,1]).reset_index(drop=True)
    return stacking_sm_pred, stacking_origin

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