#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 10:54:17 2018

@author: philip
"""
####################################
########## 구매생명주기 v2.0 ##########
####################################

### 모듈 로드
import pandas as pd
#import numpy as np


### 1. csv 파일 로드 및 전처리 
###     : 구매상품TR을 제휴사별로 나눈 csv 파일(finalA_c, finalB_c, finalC_c, finalD_c)

# 파일 로드 
final_c = pd.read_csv("/Users/philip/Workspace/Final_Project/final_c/finalA_c.csv", sep=',', encoding = "EUC-KR")
# 생략된 열 모두 보이게 하기
pd.set_option('display.expand_frame_repr', False) 
# 로드 확인
final_c.head(100) 
final_c.info() # A : 5770318 entries, 12 columns
# 결측치 확인
final_c.isnull().values.any() 
# 열별 결측치 개수 확인
final_c.isnull().sum() 
# 불필요한 열 제거
final_c = final_c.drop(['영수증번호', '대분류코드', '중분류코드', '점포코드', '구매시간', '중분류명'], axis=1)
# 고객번호, 소분류코드, 구매일자 기준으로 정렬
final_c = final_c.sort_values(['고객번호', '소분류코드', '구매일자'])
final_c


### 2. 고객/품목별 구매횟수 계산 
###     : 고객번호, 소분류코드, 구매일자를 기준으로 그룹화하여 구매일자 개수 구하기
final_c.head(10)
#final_c.groupby(['고객번호', '소분류코드'])['구매일자'].count()
#final_c.groupby(['고객번호', '소분류코드', '구매일자'])['구매일자'].count().head(100)
#buying_count = final_c.groupby(['고객번호', '소분류코드', '구매일자']).agg({'구매일자':['count']})
buying_count = final_c.groupby(['고객번호', '소분류코드']).agg({'구매일자':['count']})
# 확인
buying_count.head(100)
buying_count.info()
# MultiIndex 제거
buying_count.columns.get_level_values(0)
buying_count.columns.get_level_values(1)
buying_count.columns = ['구매횟수']

buying_count = buying_count.reset_index()
buying_count.info()
buying_count.head(100) 
buying_count.columns


### 3. 구매간격 계산
final_c.head(10)
# 처음 구매한 일자
first_buying = final_c.groupby(['고객번호', '소분류코드']).agg({'구매일자':['min']})
# 가장 최근에 구매한 일자
last_buying = final_c.groupby(['고객번호', '소분류코드']).agg({'구매일자':['max']})
# MultiIndex 제거 
first_buying.columns = ['최초구매일']
last_buying.columns = ['최근구매일']

first_buying = first_buying.reset_index()
last_buying = last_buying.reset_index()
first_buying.head(100)
last_buying.head(100)

# 고객번호와 소분류코드를 기준으로 merge
buying_days = pd.merge(first_buying, last_buying, on=['고객번호', '소분류코드'])


# 최근구매일과 최초구매일 차이 계산
from datetime import datetime
buying_days['최근-최초구매일'] = 0
d1 = buying_days['최근구매일'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
d0 = buying_days['최초구매일'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
term = d1-d0
buying_days.head(10)
term.head(10)

term = term.dt.days # 일수만 빼기
term
buying_days['최근-최초구매일'] = term
buying_days


### 4. 고객별 평균구매주기 계산
buying_days
buying_count
# 구매일과 구매횟수 데이터 merge 
buying_info = pd.merge(buying_days, buying_count, on=['고객번호', '소분류코드'])
buying_info['구매주기'] = 0
buying_info['구매주기'] = buying_info['최근-최초구매일'] // buying_info['구매횟수']
buying_info


### 5. 제품별 평균구매주기 계산
# 소분류코드로 정렬하여 보기
sorted_buying_info = buying_info.sort_values(['소분류코드'])
sorted_buying_info = sorted_buying_info.reset_index()
del sorted_buying_info['index']
sorted_buying_info
# 고객의 구매주기 중 0인 품목의 구매주기는 제거
sorted_buying_info[sorted_buying_info['구매주기'] == 0 ]
sorted_buying_info = sorted_buying_info.drop(sorted_buying_info[sorted_buying_info['구매주기'] == 0 ].index)
sorted_buying_info
# 모든 고객의 구매주기를 품목별로 평균 구하기
#sorted_buying_info['구매주기'][sorted_buying_info['소분류코드'] == 'A010101'].mean()
buying_cycle = sorted_buying_info.groupby(['소분류코드']).agg({'구매주기':['mean']})
buying_cycle
# MultiIndex 제거
buying_cycle.columns = ['구매주기']
buying_cycle = buying_cycle.reset_index()
# 열 정수화
buying_cycle['구매주기'] = buying_cycle['구매주기'].astype(int)
buying_cycle


### 6. csv 파일로 저장
buying_cycle.to_csv("/Users/philip/Workspace/Final_Project/lifecycle/lifecycle_C.csv")
