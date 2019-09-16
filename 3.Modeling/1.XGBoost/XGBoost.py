# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:04:24 2018

@author: COM
"""


### XGBoost 시행 전 Train vs Test set 나누기
###

#from matplotlib import font_manager, rc

#%% matplot 글꼴 
## 

#font_location = 'C:/Windows/Fonts/D2Coding.ttf'
#font_name = font_manager.FontProperties(fname=font_location).get_name()
#print(font_name)
#rc('font', family=font_name)

#%%
##############################
###  : 대분류코드 & 구매여부
##############################

import pandas as pd
pd.set_option('display.expand_frame_repr', False)

#dummyA_final = pd.read_csv('D:/final/dataset/competitor_A.csv', sep=',')
dummyA_final = pd.read_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/competitor_B.csv', sep=',')
dummyA_final.info()
dummyA_final[['하이마트보유', '다둥이보유', '롭스보유', '더영보유']] = dummyA_final[['하이마트보유', '다둥이보유', '롭스보유', '더영보유']].astype(int)
#dummyA_final = dummyA_final.drop(['새중분류코드'], axis=1)
dummyA_final = dummyA_final.drop(['중분류명'], axis=1)
dummyA_final = dummyA_final.drop(['소분류명'], axis=1)
dummyA_final.info()

#dummyA_final.iloc[ : ,26:27]
#dummyA_final.loc[:, 'A1':'A1']

# 기존 + 더미 붙이기
dataA_big_dummy = pd.read_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/dataB_dummy.csv', sep=',')
dummyA_final = pd.concat([dummyA_final, dataA_big_dummy], axis=1)
dummyA_final.info()






#%%
##############################
###  TRAIN SET & TEST SET 나누기
##############################
from datetime import datetime
import numpy as np

#%%
## 구매일자 date 타입으로 바꾸기
dummyA_final['구매일자']
dummyA_final['구매일자'] = dummyA_final['구매일자'].map(lambda x: x.replace('-', ''))
dummyA_final['구매일자'] = dummyA_final['구매일자'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
#%%

## 구매년도, 구매월 열 추가 : train/test set 나누는 기준
buying_year = dummyA_final['구매일자'].dt.year
buying_year = buying_year.to_frame()

buying_month = dummyA_final['구매일자'].dt.month
buying_month = buying_month.to_frame()

buying_year.columns = ['구매년도']
buying_month.columns = ['구매월'] 

buying_year
buying_month

#%%

## 기존 + 구매년도, 구매월 열 붙이기 
dummyA_final = pd.concat([dummyA_final, buying_year], axis=1)
dummyA_final = pd.concat([dummyA_final, buying_month], axis=1)

# 확인 
dummyA_final[['구매일자', '구매년도', '구매월']]
dummyA_final.info()
#dummyA_final = dummyA_final.drop(['구매년도', '구매월'], axis=1)


#%%
## 필요없는 열 제거

## 구매일자, 고객번호, 대분류코드, 중분류코드, 소분류코드, 영수증번호, 새대분류코드 빼고
## factorzing 

#dummyA_final = dummyA_final.drop(['고객번호', '구매일자', '영수증번호', 
#                                  '대분류코드', '중분류코드', '소분류코드', '새대분류코드'], axis=1)


dummyA_final = dummyA_final.drop(['구매일자', '영수증번호', 
                                  '대분류코드', '중분류코드', '소분류코드'], axis=1)

dummyA_final.columns.values


#%%
## object 타입 열 factorizing
dummyA_final['제휴사'] = pd.factorize(dummyA_final['제휴사'])[0]
dummyA_final['성별'] = pd.factorize(dummyA_final['성별'])[0]
dummyA_final['연령대'] = pd.factorize(dummyA_final['연령대'])[0]
dummyA_final['자주가는제휴사'] = pd.factorize(dummyA_final['자주가는제휴사'])[0]
dummyA_final['등급'] = pd.factorize(dummyA_final['등급'])[0]
#dummyA_final['중분류명'] = pd.factorize(dummyA_final['중분류명'])[0]
#dummyA_final['소분류명'] = pd.factorize(dummyA_final['소분류명'])[0]
dummyA_final.info()


#%%
##################################
##################################

## TRAIN & TEST SET 분리

## TEST : 2015년 11월, 12월 데이터 
test1 = dummyA_final[np.logical_and(dummyA_final['구매년도']==2015, dummyA_final['구매월']==11)]
test2 = dummyA_final[np.logical_and(dummyA_final['구매년도']==2015, dummyA_final['구매월']==12)]


## TRAIN : 나머지 
train1 = dummyA_final[dummyA_final['구매년도']==2014]
train2 = dummyA_final[np.logical_and(dummyA_final['구매년도']==2015, dummyA_final['구매월']<11)]

## 하나로 붙이기
test_final = test1.append(test2)
train_final = train1.append(train2)
test_final['고객번호']
test_cust = pd.DataFrame(test_final['고객번호'], columns=['고객번호'])
train_cust = pd.DataFrame(train_final['고객번호'], columns=['고객번호'])



## 확인
len(test_final)
len(train_final)

test_final.head(10)



#%%
## TRAINING SET에서 더미변수 따로 분리

#trainA_y = train_final.loc[ : , 'A1':'A9']
trainA_y = train_final.loc[:, train_final.columns.str.startswith('B')]
trainA_y.columns

#trainA_x = train_final.drop(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9'], axis=1)
trainA_x = train_final[train_final.columns.drop(list(train_final.filter(regex='^B')))]
trainA_x.columns


# 확인
trainA_y 
trainA_x.info()
trainA_y.info()

#%%


## Test SET에서 더미변수 따로 분리

#testA_y = test_final.loc[ : , 'A1':'A9']
#testA_x = test_final.drop(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9'], axis=1)

testA_y = test_final.loc[:, test_final.columns.str.startswith('B')]
testA_x = test_final[test_final.columns.drop(list(test_final.filter(regex='^B')))]

#testA_y 
#testA_x.info()
testA_y.info()


## 필요없는 변수 빼주기 
#trainA_x = trainA_x.drop(['중분류명', '소분류명'], axis=1) 
trainA_x = trainA_x.drop(['구매년도', '구매월'], axis=1) 
testA_x = testA_x.drop(['구매년도', '구매월'], axis=1) 

testA_y.info()

#%%

## 최종 데이터셋
#trainA_x.info()
#trainA_y.info()

#testA_x.info()
#testA_x.head()
#testA_y.info()
#testA_y.head()


#%%

#trainA_x.to_csv('D:/final/dataset/trainA_x.csv', index=False)
#trainA_y.to_csv('D:/final/dataset/trainA_y.csv', index=False)
#testA_x.to_csv('D:/final/dataset/testA_x.csv', index=False)
#testA_y.to_csv('D:/final/dataset/testA_y.csv', index=False)

trainA_x.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/trainB_x', index=False)
trainA_y.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/trainB_y', index=False)
testA_x.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/testB_x.csv', index=False)
testA_y.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/testB_y.csv', index=False)

train_cust.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/trainB_cust.csv', index=False)
test_cust.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/testB_cust.csv', index=False)

#%%

