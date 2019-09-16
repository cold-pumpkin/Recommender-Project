#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 00:07:24 2018

@author: philip
"""
import pandas as pd
import numpy as np
####### 변수생성 : 자주가는제휴사 B
#%%
########################################
#(구매정보 + 고객정보 + 상품분류 + 자주가는제휴사 + 멤버십보유 + 채널 + 모바일_클러스터) 
########################################


## '자주가는 제휴사'별로 데이터셋 쪼개기 : B

pd.set_option('display.expand_frame_repr', False) #모든 열 보이게 하기

merged_member = pd.read_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/merged_membership.csv', sep=',')
merged_member.head(10)
# 결측치 처리
merged_member = merged_member.fillna('0')

#%%
## B(롯데마트)를 자주 이용하는 고객들의 데이터만 추출
dataB = merged_member[merged_member['자주가는제휴사']=='B']

#%%

## B를 자주 이용하는 각 고객별 총 구매금액
cluster_dataB = dataB.groupby('고객번호').agg({'구매금액':'sum'})
cluster_dataB = cluster_dataB.reset_index()
len(cluster_dataB) # 7154명
cluster_dataB.columns = ['고객번호', '총구매금액']

cluster_dataB[cluster_dataB['총구매금액']==cluster_dataB['총구매금액'].max()]


#%%

## 채널이용 데이터 불러오기
channel_info = pd.read_csv("/Users/philip/Workspace/Final_Project/Data/06.채널이용.txt", sep=',', encoding = "EUC-KR")
channel_info.isnull().values.any()
channel_info['제휴사'].unique()
channel_infoB1 = channel_info[channel_info['제휴사']=='B_MOBILE/APP']
channel_infoB2 = channel_info[channel_info['제휴사']=='B_ONLINEMALL']

channel_infoB1.columns = ['고객번호', '제휴사1', '이용횟수1']
channel_infoB2.columns = ['고객번호', '제휴사2', '이용횟수2']

#%%

## B를 자주 이용하는 각 고객들의 총 구매금액 + 채널이용 데이터 merge
## A_MOBILE/APP 사용하는 고객들의 총구매금액, 이용횟수
cluster_dataB
cluster_channelB = pd.merge(cluster_dataB, channel_infoB1, on='고객번호', how='left')
cluster_channelB
cluster_channelB = pd.merge(cluster_channelB, channel_infoB2, on='고객번호', how='left')
cluster_channelB
cluster_channelB = cluster_channelB.drop(['고객번호', '제휴사1', '제휴사2'], axis=1)
cluster_channelB

cluster_channelB = cluster_channelB.fillna(0)
cluster_channelB = cluster_channelB.astype(int)
cluster_channelB['이용횟수'] = cluster_channelB['이용횟수1'] + cluster_channelB['이용횟수2']
cluster_channelB



#%%


## 클러스터링
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# KMeans 객체 생성하여 model에 저장 : 6개의 클러스터 지정

model = KMeans(n_clusters=6, algorithm='auto')
# 클러스터링을 위한 학습 , 라벨 리턴
# 실제 클러스터링 실행
model.fit(cluster_channelB) 


# 학습된 모델로 데이터를 학습된 모델에 맞춰 군집화
# -> 어느 클러스터로 군집화가 되었는지 라벨을 리턴
# predict() : 새로운 샘플을 이미 계산된 클러스터에 할당
# fit_predict() : 클러스터링과 그룹 할당(레이블링) 동시 수행 
predict = pd.DataFrame(model.predict(cluster_channelB))


cluster_channelB = pd.concat([cluster_channelB, predict], axis=1)
cluster_channelB.head(10)
cluster_channelB = cluster_channelB.drop(['이용횟수1', '이용횟수2'], axis=1)
cluster_channelB.columns = ['전체구매액', '이용횟수', '채널클러스터B']


# 시각화 - 산점도
plt.scatter(cluster_channelB['이용횟수'], cluster_channelB['전체구매액'], c=cluster_channelB['채널클러스터B'])    

# 고객번호, 전체구매액, 이용횟수, 채널클러스터A
cluster_channelB
clusteredB = pd.concat([cluster_dataB['고객번호'], cluster_channelB], axis=1)
clusteredB.head()

# 채널클러스터B 변수 추가한 csv 파일 저장 
dataB = pd.merge(dataB, clusteredB, on='고객번호', how='left') 
dataB.info()
dataB.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/channel_clusterB.csv', index=False)

#%%
########################################
# (구매정보 + 고객정보 + 상품분류 + 자주가는제휴사 + 멤버십보유 + 채널 + 클러스터 + 등급)
########################################

## 롯데 백화점 등급 구하기
from datetime import datetime

# 열 전체 Date 타입으로 바꾸기
dataB.info()
type(dataB['구매일자'][0])
dataB['구매일자'] = dataB['구매일자'].map(lambda x: x.replace('-', ''))
dataB['구매일자'] = dataB['구매일자'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))



# 등급을 계산하기 위해 직전분기(3개월) 구매 금액 
# 2015/10 ~ 2015/12 구매 관련 데이터 추출
dataB_quater = dataB.loc[dataB['구매일자'].dt.year == 2015]        
dataB_quater = dataB_quater.loc[dataB_quater['구매일자'].dt.month >= 10]  


# 구매금액 계산
dataB_sum = dataB_quater.groupby('고객번호').agg({'구매금액':'sum'})
dataB_sum = dataB_sum.reset_index()
dataB_sum
dataB_sum['등급'] = ''
dataB_sum

# 등급 계산
dataB_sum['등급'] = np.where(dataB_sum['구매금액'] >= 600000, 'Platinum', 
         np.where(np.logical_and(dataB_sum['구매금액'] >= 350000, dataB_sum['구매금액'] < 600000), 'Gold', 
                  np.where(np.logical_and(dataB_sum['구매금액'] >= 150000, dataB_sum['구매금액'] < 350000), 'Silver', 
                           np.where(dataB_sum['구매금액'] < 150000, 'Family', 'Family')))) 

len(dataB_sum.loc[dataB_sum['등급'] == 'Platinum']) + len(dataB_sum.loc[dataB_sum['등급'] == 'Gold'])+len(dataB_sum.loc[dataB_sum['등급'] == 'Silver'])+len(dataB_sum.loc[dataB_sum['등급'] == 'Family'])
len(dataB_sum.loc[dataB_sum['등급'] == 'Gold'])
len(dataB_sum.loc[dataB_sum['등급'] == 'Silver'])
len(dataB_sum.loc[dataB_sum['등급'] == 'Family'])

dataB_sum = dataB_sum.drop(['구매금액'], axis=1)
dataB_sum
pd.merge(dataB, dataB_sum, on='고객번호', how='left')

dataB = pd.merge(dataB, dataB_sum, on='고객번호', how='left')
dataB = dataB.drop(['구매금액_y'], axis=1)

dataB.info()
dataB.isna().any()
dataB[dataB['등급'].isnull()]['구매금액'].max()
len(dataB[dataB['등급'].isnull()]['구매금액'])
dataB[dataB['등급']=='Family']

# 최근 3개월 구매하지 않은 고객 -> Family
dataB.loc[dataB['등급'].isnull(),'등급'] = 'Family'


#%%

########################################
############### (구매정보 + 고객정보 + 상품분류 + 자주가는제휴사 + 멤버십보유 + 채널 + 클러스터 + 등급 + 구매시간그룹) 
########################################
# 구매시간 그룹화 

dataB['구매시간그룹화'] = 0
dataB.loc[(dataB["구매시간"] >= 0) & (dataB["구매시간"] < 2), "구매시간그룹화"] = 1
dataB.loc[(dataB["구매시간"] >= 2) & (dataB["구매시간"] < 4), "구매시간그룹화"] = 2
dataB.loc[(dataB['구매시간'] >= 4) & (dataB['구매시간'] < 6), "구매시간그룹화"] = 3
dataB.loc[(dataB['구매시간'] >= 6) & (dataB['구매시간'] < 8), "구매시간그룹화"] = 4
dataB.loc[(dataB['구매시간'] >= 8) & (dataB['구매시간'] < 10), "구매시간그룹화"] = 5
dataB.loc[(dataB['구매시간'] >= 10) & (dataB['구매시간'] < 12), "구매시간그룹화"] = 6
dataB.loc[(dataB['구매시간'] >= 12) & (dataB['구매시간'] < 14), "구매시간그룹화"] = 7
dataB.loc[(dataB['구매시간'] >= 14) & (dataB['구매시간'] < 16), "구매시간그룹화"] = 8
dataB.loc[(dataB['구매시간'] >= 16) & (dataB['구매시간'] < 18), "구매시간그룹화"] = 9
dataB.loc[(dataB['구매시간'] >= 18) & (dataB['구매시간'] < 20), "구매시간그룹화"] = 10
dataB.loc[(dataB['구매시간'] >= 20) & (dataB['구매시간'] < 22), "구매시간그룹화"] = 11
dataB.loc[dataB['구매시간'] >= 22, "구매시간그룹화"] = 12
dataB['구매시간그룹화']

dataB.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/time_groupB.csv', index=False)

#%%


dataB.info()
########################################
## 총구매금액 & 구매횟수 클러스터 변수 추가
## (구매정보 + 고객정보 + 상품분류 + 자주가는제휴사 + 멤버십보유 + 채널클러스터 + 등급 + 구매시간그룹 + 구매클러스터 )  
########################################

dataB_sum = dataB.groupby('고객번호').agg({'구매금액':'sum'})
dataB_sum = dataB_sum.reset_index()
dataB_sum.columns = ['고객번호', '총구매금액']
dataB_sum


dataB_count = dataB.groupby('고객번호').agg({'영수증번호':'count'}).reset_index()
dataB_count.columns = ['고객번호', '구매횟수']

cluster_dataB = pd.merge(dataB_count, dataB_sum, on='고객번호')
cluster_dataB = cluster_dataB.drop(['고객번호'], axis=1)
cluster_dataB

# 클러스터
model = KMeans(n_clusters=6, algorithm='auto')
# 클러스터링을 위한 학습 , 라벨 리턴
# 실제 클러스터링 실행
model.fit(cluster_dataB) 


# 학습된 모델로 데이터를 학습된 모델에 맞춰 군집화
# -> 어느 클러스터로 군집화가 되었는지 라벨을 리턴
# predict() : 새로운 샘플을 이미 계산된 클러스터에 할당
# fit_predict() : 클러스터링과 그룹 할당(레이블링) 동시 수행 
predict = pd.DataFrame(model.predict(cluster_dataB))
predict.info()
 
cluster_dataB = pd.concat([cluster_dataB, predict], axis=1)
cluster_dataB.columns = ['구매횟수', '총구매금액', '구매클러스터']
cluster_dataB
# 시각화 - 산점도
plt.scatter(cluster_dataB['구매횟수'], cluster_dataB['총구매금액'], c=cluster_dataB['구매클러스터'])    

# merge
cluster_dataB = pd.concat([dataB_sum['고객번호'], cluster_dataB], axis=1)
cluster_dataB
cluster_dataB = cluster_dataB.drop(['구매횟수', '총구매금액'], axis=1)
cluster_dataB

dataB = pd.merge(dataB, cluster_dataB, on='고객번호', how='left')
dataB.info()
dataB.isnull().any()
dataB.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/dataB_cluster.csv', index=False)



#%%


#%%


########################################
## 경쟁사 이용 변수 추가
## (구매정보 + 고객정보 + 상품분류 + 자주가는제휴사 + 멤버십보유 + 클러스터 + 등급 + 구매시간그룹 + 구매클러스터 )  
########################################

competitor_info = pd.read_csv("/Users/philip/Workspace/Final_Project/Data/04.경쟁사이용.txt", sep=',', encoding = "EUC-KR")
competitor_info.isnull().values.any() 


competitor_groupB = competitor_info.groupby(['고객번호', '제휴사', '경쟁사']).agg({'이용년월':'count'})
competitor_groupB = competitor_groupB.reset_index()
competitor_groupB.columns = ['고객번호', '제휴사', '경쟁사', '이용횟수']
competitor_groupB = competitor_groupB[competitor_groupB['제휴사']=='B']
competitor_groupB = competitor_groupB.reset_index(drop=True)

#competitor_groupA = competitor_groupA.drop('index', axis=1)
competitor_groupB

competitor_tableB = pd.pivot_table(competitor_groupB, index=['고객번호'], values='이용횟수', columns=['경쟁사'], aggfunc=np.sum)


competitor_tableB = competitor_tableB.fillna(0)
competitor_tableB = competitor_tableB.astype(int)
competitor_tableB = competitor_tableB.reset_index()
competitor_tableB

#competitor_A = competitor_groupA.groupby('고객번호').agg({'이용횟수':'sum'}).reset_index()

dataB.info()
dataB = dataB.drop(['이용횟수'], axis=1)

dataB = pd.merge(dataB, competitor_tableB, on='고객번호', how='left')
dataB.isnull().any()
dataB = dataB.fillna(0)
dataB = dataB.rename(columns={'B01':'경쟁사1이용횟수', 'B02':'경쟁사2이용횟수'})

dataB[['경쟁사1이용횟수','경쟁사2이용횟수']] = dataB[['경쟁사1이용횟수','경쟁사2이용횟수']].astype(int)
dataB[['고객번호', '경쟁사1이용횟수', '경쟁사2이용횟수']]


dataB.to_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/competitor_B.csv', index=False)
dataB.info()
#%%