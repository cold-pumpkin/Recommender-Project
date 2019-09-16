# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 17:51:05 2018

@author: COM
"""

import pandas as pd
import numpy as np

### 필터 적용전 알고리즘의 정확도 구하기


#알고리즘 돌려서 나온 고객별 상품의 예상점수가 들어있는 테이블
result_bind = pd.read_csv('D:/final/id_result_bind.csv', sep=',')


#실제 고객 영수증 - 공모전에서 제공한 데이터
purchase_origin = pd.read_csv('D:/final/제3회 Big Data Competition-분석용데이터-02.구매상품TR.txt', sep=',', engine='python')
purchase_origin.head()



### 데이터 가공하기

# 고객별로 점수가 가장 높은 상품 뽑아내기 위해 정렬
# V1 : 고객번호  V2 : 소분류코드  
result_bind = result_bind.sort_values(by=['V1', 'pred_rvec'], ascending=[True, False])
result_bind


# 중복되는 행을 제거함으로써 고객당 점수 1등의 상품만 뽑아내기 위함
tr = result_bind.drop_duplicates(subset='V1')
tr = tr.reset_index(drop=True)
tr.info()


# 열 이름 지정
tr.columns = ['고객번호', '소분류코드', '점수']


# 데이터 저장

#### 2015년 12월 데이터만 뽑기
#purchase_origin['리얼']

purchase_origin['구매일자'] = purchase_origin['구매일자'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d')) 
purchase_origin['구매일자'] = pd.to_datetime(purchase_origin['구매일자'].astype(str), format='%Y%m%d')


test = purchase_origin[np.logical_and(purchase_origin['구매일자'].dt.year == 2015,
                                      purchase_origin['구매일자'].dt.month == 12)]
test = test.reset_index(drop=True)
test.info()


# 정확도 계산에 필요한 열만 추출
test = test[['고객번호', '소분류코드']]
test.info()

# 12월에 구매이력이 존재하는 고객을 추출
test_cust = test['고객번호'].unique()
len(test_cust)  #전체 19383명중에 19119명이 12월에 구매 함



tr.info()
test.info()


# 소분류코드 -> 숫자
test['소분류코드'].str.replace('A', '9')
test['소분류코드'] = test['소분류코드'].str.replace('A', '9')
test['소분류코드'] = test['소분류코드'].str.replace('B', '8')
test['소분류코드'] = test['소분류코드'].str.replace('C', '7')
test['소분류코드'] = test['소분류코드'].str.replace('D', '6')
test['소분류코드']


# 12월 데이터와 알고리즘 적용 후 데이터의 고객번호와 소분류코드가 일치하는 행만 추출
tr['소분류코드'] = tr['소분류코드'].astype(str)
tr.info()
test.info()

fitdata = pd.merge(tr, test, on=['고객번호', '소분류코드'])
fitdata.info()


# 고객번호가 중복되게 나오므로 고객번호로 유니크한 행 추출 
onefit = fitdata.drop_duplicates(subset='고객번호')
onefit.head()
onefit.info()

# 
len(onefit)/len(test_cust)


