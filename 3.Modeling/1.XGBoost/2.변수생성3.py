#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 11:56:17 2018

@author: philip
"""

import pandas as pd
pd.set_option('display.expand_frame_repr', False) #모든 열 보이게 하기
data = pd.read_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/timegroup_data.csv", sep=',')
data.info()

#### 1. 1회 방문시 구매한 상품의 개수(한번 구매시 보통 얼만큼의 상품을 사는지 확인)
buying_count = data.groupby(['고객번호', '영수증번호']).size().reset_index()
buying_count.columns = ['고객번호', '영수증번호', '구매횟수']
buying_count.head(10)
avg_buying_count = buying_count.groupby(['고객번호']).agg({'구매횟수':'mean'})
avg_buying_count = avg_buying_count.reset_index()
avg_buying_count.columns = ['고객번호', '회당평균구매수']
avg_buying_count.head(10)    
data = pd.merge(data, avg_buying_count, on='고객번호', how='left')
data.info()



#### 2. 고객별 구매한 상품 종류 (소분류코드 기준)

code_count = data.groupby(['고객번호', '소분류코드']).size()
code_count = code_count.reset_index()
code_count.columns = ['고객번호', '소분류코드', '구매횟수']
type(code_count)
code_count.head(10)

#data.groupby(['고객번호', '소분류코드']).size()
id_result = code_count.groupby('고객번호')['소분류코드'].nunique()
id_result = id_result.reset_index()
id_result.columns = ['고객번호', '구매상품종류수']
id_result.head(10)

data = pd.merge(data, id_result, on='고객번호', how='left')
data.info()



### 구매일자 년/월/일로 잘라서 저장
from datetime import datetime
data['구매일자'][0]
#data['구매일자'].dt.date
#type(data['구매일자'][0])

data['구매일자'] = data['구매일자'].map(lambda x: x.replace('-', ''))
data['구매일자'] = data['구매일자'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
data['구매일자'][0]
type(data['구매일자'][0])

data['구매년도'] = data['구매일자'].map(lambda x: x.year)
data['구매월'] = data['구매일자'].map(lambda x: x.month)
data['구매일'] = data['구매일자'].map(lambda x: x.day)
data[['구매일자','구매년도','구매월', '구매일']].head(10)
data.info()


data.to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/final_data.csv", index=False)

