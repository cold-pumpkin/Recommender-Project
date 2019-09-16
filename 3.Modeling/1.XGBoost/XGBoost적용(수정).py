# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 20:39:38 2018

@author: COM
"""
#%% 데이터 로드
import pandas as pd
import numpy as np

#x_trainA = pd.read_csv('D:/final/dataset/trainA_x.csv', sep=',')
#y_trainA_data = pd.read_csv('D:/final/dataset/trainA_y.csv', sep=',')

#x_testA = pd.read_csv('D:/final/dataset/testA_x.csv', sep=',')
#y_testA = pd.read_csv('D:/final/dataset/testA_y.csv', sep=',')


x_trainA = pd.read_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/trainB_x', sep=',')
y_trainA_data = pd.read_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/trainB_y', sep=',')

x_testA = pd.read_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/testB_x.csv', sep=',')
y_testA = pd.read_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/testB_y.csv', sep=',')

## 고객번호 따로 빼서 저장
x_trainA_id = pd.DataFrame(x_trainA['고객번호'], columns=['고객번호'])
x_testA_id = pd.DataFrame(x_testA['고객번호'], columns=['고객번호'])
                     
x_trainA = x_trainA.drop('고객번호', axis=1)
x_testA = x_testA.drop('고객번호', axis=1)

#%%
### XGBoost 설치 
### conda에서 # conda install py-xgboost

#%%
####################################
############# XGBoost #############
###################################
'''
 Scikit-Learn 의 model_selection 서브 패키지는 교차 검증을 위해 
 전체 데이터 셋에서 트레이닝용 데이터나 테스트용 데이터를 분리해 내는 여러가지 방법을 제공
 
 데이터의 양이 충분치 않을 때, 분류기 성능측정의 통계적 신뢰도를 높이기 위해서 쓰는 방법이 
 재샘플링(resampling) 기법을 사용하는데 대표적인 방법으로는 'k-fold cross validation'(교차검증)
 
 샘플을 k로의 집단으로 나눈다. (이때, 각 집단의 mean은 비슷할 수 있도록 나눈다) 
 분류기를 k-1개의 집합으로 학습을 시키고 나머지 1개의 집합으로 test하여 
 분류기의 성능을 측정
'''
#from numpy import loadtxt
from xgboost import XGBClassifier
import xgboost as xgb
#from xgboost import plot_importance

from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import zero_one_loss

#from sklearn.metrics import mean_squared_error
#from math import sqrt



#%%
########################
######## XGBoost - Kaggle
########################



#%%
# KFold parameters
NFOLDS = 4
SEED = 0
NROWS = None

kf = KFold(n_splits = NFOLDS, shuffle=True, random_state=SEED)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


# Wrapper 객체 xg 들어감 
def get_oof(clf):
    oof_train = np.zeros((ntrain,)) # train set의 크기 만큼 0이 들어간 array 만들어 줌
    oof_test = np.zeros((ntest,))   # test set의 크기 만큼 0이 들어간 array 만들어 줌
    oof_test_skf = np.empty((NFOLDS, ntest)) # NFOLDS(4) x test set 크기 만큼의 2차원 배열 만들어 줌

    for i, (train_index, test_index) in enumerate(kf.split(x_trainA)): 
        # train set의 독립변수별로 루프 
        x_tr = x_trainA.loc[train_index] # train set 독립변수(x)
        y_tr = y_trainA.loc[train_index] # train set 종속변수(y)
        
        x_te = x_trainA.loc[test_index]  # 

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_testA)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)



xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 0,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'binary:logistic',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'nrounds': 200,
    'early_stopping_rounds': 20 
}

#%%

x_trainA.info()
x_testA.info()

#y_trainA.info()
#y_testA.info()
#%%

# x번째 학습시킬 데이터 
#y_trainA.iloc[: ,0] 
y_trainA_data.info()
#y_trainA_data.reindex_axis(sorted(y_trainA_data.columns), axis=1)
y_trainA_data = y_trainA_data.sort_index(axis=1)
y_trainA_data.columns



#%%

for i in range(0, len(y_trainA_data.columns)):
    ## 
    y_trainA = y_trainA_data.iloc[:, i] # train set의 y (더미 데이터)
    ntrain = x_trainA.shape[0]
    ntest = x_testA.shape[0]

    ## Wrapper 객체 생성 
    xg = XgbWrapper(seed=SEED, params=xgb_params)
    ### 실행 : train & test
    xg_oof_train, xg_oof_test = get_oof(xg)
    
    
    ## 테스트셋 통해 정확도 확인
    print('스코어 ' + y_testA.iloc[:, i].name +' : ' + str(roc_auc_score(y_testA.iloc[:, i], xg_oof_test)))

    ## 결과를 df로 변환하고 csv파일로 저장
    result = pd.DataFrame(xg_oof_test, columns=['result_' + y_testA.iloc[:, i].name])
    result.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/result_' + y_testA.iloc[:, i].name + '.csv', index=False)


    


#%%
## Wrapper 객체 생성 
#xg = XgbWrapper(seed=SEED, params=xgb_params)


### 실행 
### train 결과 & test 결과
#xg_oof_train, xg_oof_test = get_oof(xg)


#%%


# 확인
xg_oof_train # train 결과
xg_oof_test  # test 결과


#%%

# MSE : 실제값(관측값)과 추정값과의 차이, 잔차가 얼마인지 알려주는 척도
#print("XG-CV: {}".format(sqrt(mean_squared_error(y_trainA, xg_oof_train))))
#%%

##### test set의 정답과 비교해보기 
#y_testA.iloc[:,6]
                    # test set의 정답 넣어주기 !!
#print(roc_auc_score(y_testA.iloc[:, 0], xg_oof_test))

#%%


# Train set 결과 합치기

#result_A1 = pd.DataFrame(xg_oof_test, columns=['result_A1'])
#result_A1

#result_A2 = pd.DataFrame(xg_oof_test, columns=['result_A2'])
#result_A2


#result_A3 = pd.DataFrame(xg_oof_test, columns=['result_A3'])
#result_A3


#result_A4 = pd.DataFrame(xg_oof_test, columns=['result_A4'])
#result_A4

#result_A5 = pd.DataFrame(xg_oof_test, columns=['result_A5'])
#result_A5

#result_A6 = pd.DataFrame(xg_oof_test, columns=['result_A6'])
#result_A6

#result_A7 = pd.DataFrame(xg_oof_test, columns=['result_A7'])
#result_A7


#result_A8 = pd.DataFrame(xg_oof_test, columns=['result_A8'])
#result_A8
#%% 


#result_A1.to_csv('D:/final/dataset/result_A1.csv', index=False)
#result_A2.to_csv('D:/final/dataset/result_A2.csv', index=False)
#result_A3.to_csv('D:/final/dataset/result_A3.csv', index=False)
#result_A4.to_csv('D:/final/dataset/result_A4.csv', index=False)
#result_A5.to_csv('D:/final/dataset/result_A5.csv', index=False)
#result_A6.to_csv('D:/final/dataset/result_A6.csv', index=False)
#result_A7.to_csv('D:/final/dataset/result_A7.csv', index=False)
#result_A8.to_csv('D:/final/dataset/result_A8.csv', index=False)


type(xg_oof_test)



##%% X