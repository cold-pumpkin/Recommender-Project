# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 09:05:31 2018

@author: COM
"""


### 총구매금액 & 구매횟수 클러스터 변수 추가
### (구매정보 + 고객정보 + 상품분류 + 자주가는제휴사 
### + 멤버십보유 + 채널 + 채널클러스터 + 등급 + 구매시간그룹 + 등급 + 구매시간그룹화 + 그매클러스터 + 이용횟수

#%%

import pandas as pd
pd.set_option('display.expand_frame_repr', False)


#dataA = pd.read_csv('D:/final/dataset/competitor_A.csv', sep=',')
dataA = pd.read_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/competitor_B.csv', sep=',')
dataA.info()
## 불러올 때 int가 float이 됨...

dataA[['하이마트보유', '다둥이보유', '롭스보유', '더영보유']] = dataA[['하이마트보유', '다둥이보유', '롭스보유', '더영보유']].astype(int)
#dataA = dataA.drop(['이용횟수'], axis=1)



#%%

#################################################
####### 더미변수 : 중분류코드 & 구매여부 XXX ###########
#################################################


## 새중분류코드 열 생성 : (구매한)제휴사 + (구매한)중분류코드


dataA['새중분류코드'] = ''
dataA['새중분류코드'] = dataA[['제휴사', '중분류코드']].apply(lambda x:'%s%s' % (x['제휴사'], x['중분류코드']),axis=1)
dataA['새중분류코드']

dataA_dummy = dataA[['고객번호', '새중분류코드']]
dataA_dummy.info()
dataA_dummy



dataA_dummy['이용횟수'] = 1
dataA_dummy['인덱스'] = dataA_dummy.index
dataA_dummy

dataA_dummy_table = dataA_dummy.drop(['고객번호'], axis=1)
dataA_dummy_table


dataA_dummy_table = pd.pivot_table(dataA_dummy_table, values='이용횟수', index=['인덱스'], columns=['새중분류코드'], aggfunc=np.sum)



dataA_dummy = dataA_dummy.fillna(0) 
dataA_dummy.columns.values 
col = dataA_dummy.columns 
dataA_dummy = dataA_dummy.astype(int)
dataA_dummy 
type(dataA_dummy)

#%%

###############
### 더미변수 : 대분류코드 & 구매여부
###############
import numpy as np

dataA['새대분류코드'] = ''
dataA['새대분류코드'] = dataA[['제휴사', '대분류코드']].apply(lambda x:'%s%s' % (x['제휴사'], x['대분류코드']),axis=1)
#dataA['새대분류코드']

dataA_big = dataA[['고객번호', '새대분류코드']]
dataA_big['인덱스'] = dataA_big.index
dataA_big['이용횟수'] = 1

#dataA_big

dataA_dummy_table = pd.pivot_table(dataA_big, values='이용횟수', index=['인덱스'], columns=['새대분류코드'], aggfunc=np.sum)
#dataA_dummy_table.columns


#dataA_dummy_table.filter(regex='B')
#dataA_big_dummy = dataA_dummy_table.iloc[: ,:9]

dataA_big_dummy = dataA_dummy_table.loc[:, dataA_dummy_table.columns.str.startswith('B')]
dataA_big_dummy = dataA_big_dummy.fillna(0)
dataA_big_dummy = dataA_big_dummy.astype(int)
#dataA_big_dummy 

#dataA_big_dummy.columns.values


#dataA_big_final = pd.concat([dataA, dataA_big_dummy], axis=1)
#dataA_big_final.head(10)

dataA_big_dummy.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/dataB_dummy.csv', index=False)
#dataA_big_final.to_csv('D:/final/dataset/dataB_dummy.csv', index=False)

#%%

