#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:42:31 2018

@author: philip
"""

#%%
########################################
############### (구매정보 + 고객정보 + 상품분류 + 자주가는제휴사 + 멤버십보유 + 채널 + 모바일_클러스터) 
########################################


## '자주가는 제휴사'별로 데이터셋 쪼개기
import pandas as pd
pd.set_option('display.expand_frame_repr', False) #모든 열 보이게 하기
merged_member = pd.read_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/merged_membership.csv', sep=',')
merged_member.head(10)
merged_member = merged_member.fillna('0')
#%%

## A(롯데백화점)를 자주 이용하는 고객들의 데이터만 추출
dataA = merged_member[merged_member['자주가는제휴사']=='A']
dataA[dataA['고객번호']==15999]
#%%


## A를 자주 이용하는 각 고객들의 총 구매금액
cluster_dataA = dataA.groupby('고객번호').agg({'구매금액':'sum'})
cluster_dataA = cluster_dataA.reset_index()
cluster_dataA.columns = ['고객번호', '총구매금액']
cluster_dataA[cluster_dataA['총구매금액']==cluster_dataA['총구매금액'].max()]
cluster_dataA[cluster_dataA['고객번호']==16406]
#%%

## 채널이용 데이터 불러오기
channel_info = pd.read_csv("/Users/philip/Workspace/Final_Project/Data/06.채널이용.txt", sep=',', encoding = "EUC-KR")
channel_info.isnull().values.any()
channel_info.head()
channel_infoA = channel_info[channel_info['제휴사']=='A_MOBILE/APP']

#%%

#%%
## A를 자주 이용하는 각 고객들의 총 구매금액 + 채널이용 데이터 merge
## A_MOBILE/APP 사용하는 고객들의 총구매금액, 이용횟수
cluster_channelA = pd.merge(cluster_dataA, channel_infoA, on='고객번호', how='left')
cluster_channelA = cluster_channelA.drop(['고객번호', '제휴사'], axis=1)
cluster_channelA = cluster_channelA.fillna(0)
cluster_channelA = cluster_channelA.astype(int)
cluster_channelA




#%%
## 정규화

from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# 전체구매액, 구매횟수 하나로 붙여서 정규화
#cluster_k.iloc[:,0]
#cluster_k.iloc[:,1]
#values = cluster_k.iloc[:,0].append(cluster_k.iloc[:,1])
#scaled_values = minmax_scale(values, feature_range=(0, 1))
#len(scaled_values)//2


#scaled_channelA = minmax_scale(cluster_channelA, feature_range=(0,1))
#scaled_channelA
#pd.DataFrame(scaled_channelA, columns=['총구매금액', '이용횟수'])



#scaled_values1 = pd.DataFrame(scaled_values[:len(scaled_values)//2], columns=['전체구매액'])
#scaled_values2 = pd.DataFrame(scaled_values[len(scaled_values)//2+1:], columns=['구매횟수'])
#scaled_values = pd.concat([scaled_values1.reset_index(drop=True), scaled_values2], axis=1)
#scaled_values


#%%
## 클러스터링
#cluster_channelA = cluster_channelA.drop(['채널클러스터A'], axis=1)
cluster_channelA
# KMeans 객체 생성하여 model에 저장 : 6개의 클러스터 지정
model = KMeans(n_clusters=6, algorithm='auto')
# 클러스터링을 위한 학습 , 라벨 리턴
# 실제 클러스터링 실행
model.fit(cluster_channelA) 


# 학습된 모델로 데이터를 학습된 모델에 맞춰 군집화
# -> 어느 클러스터로 군집화가 되었는지 라벨을 리턴
# predict() : 새로운 샘플을 이미 계산된 클러스터에 할당
# fit_predict() : 클러스터링과 그룹 할당(레이블링) 동시 수행 
predict = pd.DataFrame(model.predict(cluster_channelA))
#predict.info()



cluster_channelA = pd.concat([cluster_channelA, predict], axis=1)
cluster_channelA.head(10)
cluster_channelA.columns = ['전체구매액', '이용횟수', '채널클러스터A']


# 시각화 - 산점도
plt.scatter(cluster_channelA['이용횟수'], cluster_channelA['전체구매액'], c=cluster_channelA['채널클러스터A'])    

# 고객번호, 전체구매액, 이용횟수, 채널클러스터A
clusteredA = pd.concat([cluster_dataA['고객번호'], cluster_channelA], axis=1)
clusteredA.head()

# 채널클러스터A 변수 추가한                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            csv 파일 저장 
dataA = pd.merge(dataA, clusteredA, on='고객번호', how='left') 
dataA.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/channel_clusterA.csv', index=False)



### 자주가는 제휴사별로 데이터셋 나누어 저장
#data[data['자주가는제휴사']=='A'].to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/feq_companyA.csv", index=False)
#data[data['자주가는제휴사']=='B'].to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/feq_companyB.csv", index=False)
#data[data['자주가는제휴사']=='C'].to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/feq_companyC.csv", index=False)
#data[data['자주가는제휴사']=='D'].to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/feq_companyD.csv", index=False)



#%%
########################################
############### (구매정보 + 고객정보 + 상품분류 + 자주가는제휴사 + 멤버십보유 + 채널클러스터 + 등급) 
########################################

## 롯데 백화점 등급 구하기

from datetime import datetime

# 열 전체 Date 타입으로 바꾸기
dataA.info()
type(dataA['구매일자'][0])
dataA['구매일자'] = dataA['구매일자'].map(lambda x: x.replace('-', ''))
dataA['구매일자'] = dataA['구매일자'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))

# 등급을 계산하기 위해
# 2014/12 ~ 2015/11 구매 관련 데이터 추출
dataA_2014 = dataA.loc[dataA['구매일자'].dt.year == 2014]         
dataA_201412 = dataA_2014.loc[dataA_2014['구매일자'].dt.month == 12]
dataA_201412

dataA_2015 = dataA.loc[dataA['구매일자'].dt.year == 2015]        
dataA_2015 = dataA_2015.loc[dataA_2015['구매일자'].dt.month <= 11]  
dataA_2015.info()

# 해당 데이터 합치기
dataA_grade = dataA_201412.append(dataA_2015)
dataA_grade

# 구매금액 계산
dataA_sum = dataA_grade.groupby('고객번호').agg({'구매금액':'sum'})
dataA_sum = dataA_sum.reset_index()
dataA_sum['등급'] = ''
dataA_sum


len(dataA_sum.loc[dataA_sum['구매금액'] >= 100000000])

import numpy as np

dataA.info()

dataA_sum['등급'] = np.where(dataA_sum['구매금액'] >= 100000000, 'LENITH', 
         np.where(np.logical_and(dataA_sum['구매금액'] >= 60000000, dataA_sum['구매금액'] < 100000000), 'MVG-Prestige', 
                  np.where(np.logical_and(dataA_sum['구매금액'] >= 40000000, dataA_sum['구매금액'] < 60000000), 'MVG-Crown', 
                           np.where(np.logical_and(dataA_sum['구매금액'] >= 18000000, dataA_sum['구매금액'] < 40000000), 'MVG-Ace', 
                                    np.where(dataA_sum['구매금액'] < 18000000, 'Ordinary', 'Ordinary')))))
         
         

dataA_sum
len(dataA_sum.loc[dataA_sum['등급'] == 'LENITH'])
len(dataA_sum.loc[dataA_sum['등급'] == 'MVG-Prestige'])
len(dataA_sum.loc[dataA_sum['등급'] == 'MVG-Crown'])
len(dataA_sum.loc[dataA_sum['등급'] == 'MVG-Ace'])
len(dataA_sum.loc[dataA_sum['등급'] == 'Ordinary'])
dataA_sum = dataA_sum.drop(['구매금액'], axis=1)
dataA_sum.isna().any()
#len(dataA_sum.loc[dataA_sum['구매금액'] >= 100000000])
#len(dataA_sum.loc[np.logical_and(dataA_sum['구매금액'] >= 60000000, dataA_sum['구매금액'] < 100000000), : ])
#len(dataA_sum.loc[np.logical_and(dataA_sum['구매금액'] >= 40000000, dataA_sum['구매금액'] < 60000000), : ])
#len(dataA_sum.loc[np.logical_and(dataA_sum['구매금액'] >= 18000000, dataA_sum['구매금액'] < 40000000), : ])
#len(dataA_sum.loc[dataA_sum['구매금액'] < 18000000, : ])


dataA = pd.merge(dataA, dataA_sum, on='고객번호', how='left')
dataA.isna().any()
dataA['등급'].head(10000)

dataA.loc[:, dataA.isna().any()]


dataA = dataA.drop(['전체구매액', '이용횟수'], axis=1)
dataA.info()



#%%

########################################
############### (구매정보 + 고객정보 + 상품분류 + 자주가는제휴사 + 멤버십보유 + 채널 + 클러스터 + 등급 + 구매시간그룹) 
########################################
# 구매시간 그룹화 

dataA['구매시간그룹화'] = 0
dataA.loc[(dataA["구매시간"] >= 0) & (dataA["구매시간"] < 2), "구매시간그룹화"] = 1
dataA.loc[(dataA["구매시간"] >= 2) & (dataA["구매시간"] < 4), "구매시간그룹화"] = 2
dataA.loc[(dataA['구매시간'] >= 4) & (dataA['구매시간'] < 6), "구매시간그룹화"] = 3
dataA.loc[(dataA['구매시간'] >= 6) & (dataA['구매시간'] < 8), "구매시간그룹화"] = 4
dataA.loc[(dataA['구매시간'] >= 8) & (dataA['구매시간'] < 10), "구매시간그룹화"] = 5
dataA.loc[(dataA['구매시간'] >= 10) & (dataA['구매시간'] < 12), "구매시간그룹화"] = 6
dataA.loc[(dataA['구매시간'] >= 12) & (dataA['구매시간'] < 14), "구매시간그룹화"] = 7
dataA.loc[(dataA['구매시간'] >= 14) & (dataA['구매시간'] < 16), "구매시간그룹화"] = 8
dataA.loc[(dataA['구매시간'] >= 16) & (dataA['구매시간'] < 18), "구매시간그룹화"] = 9
dataA.loc[(dataA['구매시간'] >= 18) & (dataA['구매시간'] < 20), "구매시간그룹화"] = 10
dataA.loc[(dataA['구매시간'] >= 20) & (dataA['구매시간'] < 22), "구매시간그룹화"] = 11
dataA.loc[dataA['구매시간'] >= 22, "구매시간그룹화"] = 12
dataA['구매시간그룹화']

dataA.groupby('구매시간그룹화').size()

dataA.info()
dataA = dataA.fillna('Ordinary')

dataA.head()
dataA.isnull().any()
dataA.isnull().sum()


dataA.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/time_groupA.csv', index=False)

#%%


########################################
## 총구매금액 & 구매횟수 클러스터 변수 추가
## (구매정보 + 고객정보 + 상품분류 + 자주가는제휴사 + 멤버십보유 + 채널 + 클러스터 + 등급 + 구매시간그룹 + 구매클러스터 )  
########################################

dataA_sum = dataA.groupby('고객번호').agg({'구매금액':'sum'})
dataA_sum = dataA_sum.reset_index()
dataA_sum.columns = ['고객번호', '총구매금액']
dataA_sum


dataA
dataA_count = dataA.groupby('고객번호').agg({'영수증번호':'count'}).reset_index()
dataA_count.columns = ['고객번호', '구매횟수']

cluster_dataA = pd.merge(dataA_count, dataA_sum, on='고객번호')
cluster_dataA = cluster_dataA.drop(['고객번호'], axis=1)
cluster_dataA

# 클러스터
model = KMeans(n_clusters=6, algorithm='auto')
# 클러스터링을 위한 학습 , 라벨 리턴
# 실제 클러스터링 실행
model.fit(cluster_dataA) 


# 학습된 모델로 데이터를 학습된 모델에 맞춰 군집화
# -> 어느 클러스터로 군집화가 되었는지 라벨을 리턴
# predict() : 새로운 샘플을 이미 계산된 클러스터에 할당
# fit_predict() : 클러스터링과 그룹 할당(레이블링) 동시 수행 
predict = pd.DataFrame(model.predict(cluster_dataA))
predict.info()
 
cluster_dataA = pd.concat([cluster_dataA, predict], axis=1)
cluster_dataA.columns = ['구매횟수', '총구매금액', '구매클러스터']
cluster_dataA
# 시각화 - 산점도
plt.scatter(cluster_dataA['구매횟수'], cluster_dataA['총구매금액'], c=cluster_dataA['구매클러스터'])    

# merge
cluster_dataA = pd.concat([dataA_sum['고객번호'], cluster_dataA], axis=1)
cluster_dataA
cluster_dataA = cluster_dataA.drop(['구매횟수', '총구매금액'], axis=1)
cluster_dataA

dataA = pd.merge(dataA, cluster_dataA, on='고객번호', how='left')
dataA.info()
dataA.isnull().any()
dataA.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/dataA_cluster.csv', index=False)


# 고객번호, 전체구매액, 이용횟수, 채널클러스터A
cluster_dataA = pd.concat([])
clusteredA = pd.concat([cluster_dataA['고객번호'], cluster_channelA], axis=1)
clusteredA.head()



#%%


########################################
## 경쟁사 이용 변수 추가
## (구매정보 + 고객정보 + 상품분류 + 자주가는제휴사 + 멤버십보유 + 채널 + 클러스터 + 등급 + 구매시간그룹 + 구매클러스터 )  
########################################

competitor_info = pd.read_csv("/Users/philip/Workspace/Final_Project/Data/04.경쟁사이용.txt", sep=',', encoding = "EUC-KR")
competitor_info.isnull().values.any() 
competitor_info[competitor_info['고객번호']==1]

competitor_groupA = competitor_info.groupby(['고객번호', '제휴사', '경쟁사']).agg({'이용년월':'count'})
competitor_groupA = competitor_groupA.reset_index()
competitor_groupA.columns = ['고객번호', '제휴사', '경쟁사', '이용횟수']
competitor_groupA = competitor_groupA[competitor_groupA['제휴사']=='A']
competitor_groupA = competitor_groupA.reset_index(drop=True)

#competitor_groupA = competitor_groupA.drop('index', axis=1)
competitor_groupA

competitor_tableA = pd.pivot_table(competitor_groupA, index=['고객번호'], values='이용횟수', columns=['경쟁사'], aggfunc=np.sum)

competitor_tableA = competitor_tableA.fillna(0)
competitor_tableA = competitor_tableA.astype(int)
competitor_tableA = competitor_tableA.reset_index()
competitor_tableA = competitor_tableA.reset_index()
competitor_tableA

#competitor_A = competitor_groupA.groupby('고객번호').agg({'이용횟수':'sum'}).reset_index()

dataA.info()
dataA = dataA.drop(['이용횟수_x', '이용횟수_y'], axis=1)

dataA = pd.merge(dataA, competitor_tableA, on='고객번호', how='left')
dataA.isnull().any()
dataA = dataA.fillna(0)
dataA = dataA.rename(columns={'A01':'경쟁사1이용횟수', 'A02':'경쟁사2이용횟수'})

dataA[['경쟁사1이용횟수','경쟁사2이용횟수']] = dataA[['경쟁사1이용횟수','경쟁사2이용횟수']].astype(int)
dataA[['고객번호', '경쟁사1이용횟수', '경쟁사2이용횟수']]


dataA.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/competitor_A.csv', index=False)
dataA.info()
#%%