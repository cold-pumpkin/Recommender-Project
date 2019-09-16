#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 14:53:06 2018

@author: philip
"""
import pandas as pd
pd.set_option('display.expand_frame_repr', False) #모든 열 보이게 하기

data1 = pd.read_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/classification_data.csv", sep=',')
data1.info()
####### 1.중분류코드에 제휴사를 붙인 새로운 열 추가

data1['새중분류코드'] = data1['제휴사'] + data1['중분류코드'].map(str)
data1['새대분류코드'] = data1['제휴사'] + data1['대분류코드'].map(str)





####### 2.Y변수 선택(제휴사 선택)을 위한 고객별 자주 가는 계열사 찾기(최근 기준)
from datetime import datetime

# date 객체화
data1['구매일자'] = data1['구매일자'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
data1 = data1.sort_values(['고객번호', '구매일자'], ascending=[True, False])
data1 = data1.reset_index(drop=True)
data1.head(10)

# 고객별로 제휴사를 count하고 구매일자로 정렬
count_company = data1.groupby(['고객번호', '구매일자', '제휴사']).size().reset_index(name='구매품목수')
count_company = count_company.sort_values(['고객번호', '구매일자'], ascending=[True, False])
count_company = count_company.reset_index(drop=True)
#del count_company['index']
#count_company.tail(100)
#type(count_company)



# 자주가는 계열사 구하기 
#   : 고객별로 최근 1/3 기간안에 가장 자주 이용한 제휴사
#count_company.head(10)
count_company.groupby('고객번호').size()
#count_company_r = pd.read_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/count_company.csv", sep=',')
#count_company.equals(count_company_r)
#count_company_r.head(200)
#count_company.to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/count_company_py.csv", index=False)

def frequent_company(x):
    y = x.size//3
    return x.iloc[:y, 2].value_counts().idxmax() 


result = count_company.groupby('고객번호').apply(frequent_company)
result_id = pd.DataFrame(result, columns=['자주가는제휴사']).reset_index()
result_id.head(10)
result_id.info()
result_id.isnull().values.any() 


# 확인
len(result_id[result_id['자주가는제휴사']=='A'])
len(result_id[result_id['자주가는제휴사']=='B'])
len(result_id[result_id['자주가는제휴사']=='C'])
len(result_id[result_id['자주가는제휴사']=='D'])


#### 시각화
import matplotlib.pyplot as plt

labels = ['A', 'B', 'C', 'D']


## 자주가는 제휴사가 A / 제휴사 A에서 구매

data1[data1['제휴사'] == 'A'] & data1[data1['자주가는제휴사'] == 'A']
len(data1[data1['제휴사'] == 'A'])
len(data1[data1['제휴사'] == 'B'])
len(data1[data1['제휴사'] == 'C'])
len(data1[data1['제휴사'] == 'D'])

A = data1[data1['제휴사'] == 'A']
B = data1[data1['제휴사'] == 'B']
C = data1[data1['제휴사'] == 'C']
D = data1[data1['제휴사'] == 'D']


A_a = len(A[A['자주가는제휴사']=='A']) / len(A)
A_b = len(A[A['자주가는제휴사']=='B']) / len(A)
A_c = len(A[A['자주가는제휴사']=='C']) / len(A)
A_d = len(A[A['자주가는제휴사']=='D']) / len(A)


labels = ['A('+str(round(A_a*100))+'%)', 'B('+str(round(A_b*100))+'%)', 'C('+str(round(A_c*100))+'%)', 'D('+str(round(A_d*100))+'%)']

sizes = [A_a, A_b, A_c, A_d]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()







# 변수생성(최근 이용한 계열사 변수 생성, 기존 데이터에 merge)
data1.head(10)
data1.info()
data1 = pd.merge(data1, result_id, on=['고객번호'], how='left')
data.isna().any() # 결측치 없음
data.head(10)




####### 3.전체구매액 + 구매횟수로 고객 클러스터링
data1 = data

data1.head(10)
# 고객별 전체구매액
cluster_data1 = data1.groupby('고객번호').agg({'구매금액':'sum'})
cluster_data1 = cluster_data1.reset_index()
# 고객별, 영수증번호별 품목 카운트 -> ??
cnt = data1.groupby(['고객번호', '영수증번호']).size()
cnt = cnt.reset_index()
cnt.columns = ['고객번호', '영수증번호', '구매품목수']
cluster_data1.info()
cnt.info()
# 고객별로 구매횟수 카운트 
cluster_data2 = data1.groupby('고객번호').size()
cluster_data2 = cluster_data2.reset_index()
cluster_data2.columns = ['고객번호', '구매횟수']
cluster_data1.head(10)
cluster_data2.head(10)
cluster = pd.merge(cluster_data1, cluster_data2, on='고객번호')
cluster.columns = ['고객번호', '전체구매액', '구매횟수']
cluster.isna().any()



## 클러스터링 알고리즘 적용(kmeans)
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#import seaborn as sns

# 전체구매액, 구매횟수
cluster_k = cluster.drop(['고객번호'], axis=1)
cluster_k.head(10)
cluster_k.head(10)

########### 정규화 ver 1.0 (X) : 열별로 하면 안됨
############################################
############################################
scaled_data = minmax_scale(cluster_k, feature_range=(0, 1))
scaled_data = pd.DataFrame(scaled_data, columns=['전체구매액', '구매횟수'])
scaled_data



########## 정규화 RE ##########
#############################
#############################
# 전체구매액, 구매횟수 하나로 붙여서 정규화
cluster_k.iloc[:,0]
cluster_k.iloc[:,1]
values = cluster_k.iloc[:,0].append(cluster_k.iloc[:,1])
scaled_values = minmax_scale(values, feature_range=(0, 1))
len(scaled_values)//2

scaled_values1 = pd.DataFrame(scaled_values[:len(scaled_values)//2], columns=['전체구매액'])
scaled_values2 = pd.DataFrame(scaled_values[len(scaled_values)//2+1:], columns=['구매횟수'])
scaled_values = pd.concat([scaled_values1.reset_index(drop=True), scaled_values2], axis=1)
scaled_values


# KMeans 객체 생성하여 model에 저장 : 6개의 클러스터 지정
model = KMeans(n_clusters=6, algorithm='auto')

# 클러스터링을 위한 학습 , 중심점 6개 추출
model.fit(scaled_data) # 실제 클러스터링 실행

# 학습된 모델로 데이터를 학습된 모델에 맞춰 군집화
# -> 어느 클러스터로 군집화가 되었는지 라벨을 리턴
# predict() : 새로운 샘플을 이미 계산된 클러스터에 할당
# fit_predict() : 클러스터링과 그룹 할당(레이블링) 동시 수행 
predict = pd.DataFrame(model.predict(scaled_data))
predict.info()
cluster_k.info()

clustered = pd.concat([scaled_data, predict], axis=1)
clustered.head(10)
clustered.columns = ['전체구매액', '구매횟수', '클러스터']

# 시각화 - 산점도
plt.scatter(clustered['구매횟수'], clustered['전체구매액'], c=clustered['클러스터'])    

# [고객번호, 클러스터 라벨] 
cluster['고객번호']
clustered['클러스터']
cluster_final = pd.DataFrame({'고객번호':cluster['고객번호'], '고객클러스터':clustered['클러스터']})
cluster_final

# 기존 데이터와 merge
data = pd.merge(data, cluster_final, on='고객번호', how='left')
data.info()

# + 고객클러스터 csv파일 저장
data.to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/cluster_data.csv", index=False)

data.info()
