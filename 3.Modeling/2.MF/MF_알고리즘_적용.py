# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:26:57 2018

@author: COM
"""


import pandas as pd
import numpy as np

### 데이터 읽어오기
original_data = pd.read_csv('C:/Users/COM/Desktop/Lpoint/제3회 Big Data Competition-분석용데이터-02.구매상품TR.txt', sep=',', engine='python')
original_data.head()


### 구매금액을 고객번호별로 합산

sum_data = original_data.groupby('고객번호').agg({'구매금액':'sum'})

sum_data = sum_data.reset_index()
sum_data = sum_data.rename(columns={'구매금액':'총구매금액'})
sum_data


### 고객별 총구매금액 상위 25%, 50%, 75% 기준으로 나누기
sum_data.describe()
sum_data1 = sum_data[sum_data['총구매금액'] > 39349999]
sum_data2 = sum_data[np.logical_and(sum_data['총구매금액'] > 10929999, 
                                    sum_data['총구매금액'] < 39350000)]
sum_data3 = sum_data[sum_data['총구매금액'] < 10929999]

sum_data1.info() #상위 25% : 4848
sum_data2.info() #중위 50% : 9692
sum_data3.info() #하위 25% : 4843


### 각 그룹의 고객들의 정보 취합
# 고객정보만 빼기
type(sum_data1['고객번호'])
sum_data2['고객번호']
sum_data3['고객번호']
data1_cust = sum_data1['고객번호']
data2_cust = sum_data2['고객번호']
data3_cust = sum_data3['고객번호']
# 해당 고객의 구매 정보 빼오기
final_data1 = original_data.loc[original_data['고객번호'].isin(data1_cust)]
final_data2 = original_data.loc[original_data['고객번호'].isin(data2_cust)]
final_data3 = original_data.loc[original_data['고객번호'].isin(data3_cust)]
final_data1.info()
final_data2.info()
final_data3.info()
# 고객번호, 소분류코드, 구매일자만 남기기
final_data1 = final_data1[['고객번호', '소분류코드', '구매일자']]
final_data2 = final_data2[['고객번호', '소분류코드', '구매일자']]
final_data3 = final_data3[['고객번호', '소분류코드', '구매일자']]

final_data1 = final_data1.reset_index(drop=True)
final_data2 = final_data2.reset_index(drop=True)
final_data3 = final_data3.reset_index(drop=True)
# 구매일자를 날짜 형식으로 바꾸기
from datetime import datetime

final_data1['구매일자'] = final_data1['구매일자'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
final_data2['구매일자'] = final_data2['구매일자'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
final_data3['구매일자'] = final_data3['구매일자'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))


### 연령대와 성별로 고객군을 나누어 모델을 적용하는 Hierachical Modeling 수행 




# 중위 50% 데이터 적용
casting_data2 = final_data2[['고객번호','소분류코드']] 
casting_data2['value'] = 1
casting_data2

# 데이터 행렬전환 (각 상품코드를 열로 변환 후 빈도 카운트)
casted_data = pd.pivot_table(casting_data2, values='value', index='고객번호', columns=['소분류코드'], aggfunc=np.sum)
# 결측치 처리
casted_data = casted_data.fillna(0)
# float -> int
casted_data = casted_data.astype(int)
casted_data.info()
casted_data.head()
#casted_data['D010101']
#casting_data2[np.logical_and(casting_data2['고객번호']==4, casting_data2['소분류코드']=='A010101')]

# 열로되어 있는 상품명을 다시 하나의 열로 반환하고 Value(빈도수) 계산한 열과 함께 저장
casted_data = casted_data.reset_index()
recasting_data = pd.melt(casted_data, id_vars=['고객번호'])
recasting_data.info()
# 결측치 확인
recasting_data.isnull().any()
####################################
## 
#################################### 
# R에서 na행 삭제하면 나오는 것
recasting_data2 = casting_data2.groupby(['고객번호', '소분류코드']).agg({'value':'sum'})
recasting_data2 = recasting_data2.reset_index()





##################################

## test set 생성
final_data2[['고객번호', '소분류코드']]

final_data2['고객번호']
final_data2['소분류코드']

cust = final_data2['고객번호'].unique()
pd.DataFrame(cust)
goods = final_data2['소분류코드'].unique()

## 중위 50%의 고객번호와 구매한 모든 소분류코드
cust_temp = pd.DataFrame(cust, columns=['고객번호'])
goods_temp = pd.DataFrame(goods,  columns=['소분류코드'])

cust_temp['tmp'] = 1
goods_temp['tmp'] = 1

test = pd.merge(cust_temp, goods_temp, on=['tmp'])
test.info()

#%%

### Apply Matrix Factorization

# test set, train set 불러오기
#train_set = pd.read_csv('D:/final/trainset.txt', sep=',', header=None)
train_set = pd.read_csv('/Users/philip/Workspace/Final_Project/R_to_Python/56789/3.Modeling/2.MF/trainset.txt', sep=' ', encoding='EUC-KR')
train_set.info()
train_set.head()
train_set.tail()


#test_set = pd.read_csv('D:/final/testset.txt', sep=',', header=None)
test_set = pd.read_csv('D:/final/testset.txt', sep=',', header=None)
test_set.info()
test_set.head()




#%%
import numpy as np
# Matrix Factorization via Singular Value Decomposition

train_pivot = train_set.pivot(index='고객번호', columns='소분류코드', values='value').fillna(0)
train_pivot = train_pivot.astype(int)
train_pivot.head()


train_matrix = train_pivot.as_matrix()
train_matrix
cust_buying_mean = np.mean(train_matrix, axis=1)


# 
train_matrix
train_demeaned = train_matrix - cust_buying_mean.reshape(-1, 1)


# Singular Value Decomposition
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(train_demeaned, k = 50)

sigma = np.diag(sigma)


# Making Predictions from the Decomposed Matrices
predicted_buying = np.dot(np.dot(U, sigma), Vt) + cust_buying_mean.reshape(-1, 1)
preds_df = pd.DataFrame(predicted_buying, columns = train_pivot.columns)
preds_df





#%%

X = train_matrix
from sklearn.decomposition import NMF
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_

nR = np.dot(W,H)
pd.DataFrame(nR, columns=train_pivot.columns)

#%%