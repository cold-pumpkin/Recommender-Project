#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:57:26 2018

@author: philip
"""
#################################################################
#################### Matrix Factorization #######################
#################################################################


## 데이터 읽어오기
import pandas as pd 
import numpy as np


original_data = pd.read_csv("/Users/philip/Workspace/Final_Project/Data/02.구매상품TR.csv", sep=',')
original_data.info()


## 구매금액을 고객번호별로 합산 

original_data.head()

sum_data = original_data.groupby('고객번호').agg({'구매금액':'sum'})

sum_data = sum_data.reset_index()
sum_data.columns = ['고객번호', '총구매금액']

## 총구매금액 상위 25%, 50%, 75%의 기준으로 나누기
sum_data.describe()

sum_data1 = sum_data[sum_data['총구매금액'] > 39349999]
sum_data2 = sum_data[np.logical_and(sum_data['총구매금액'] > 10929999, sum_data['총구매금액'] < 39350000)]
sum_data4 = sum_data[sum_data['총구매금액'] < 10929999]

len(sum_data1)
len(sum_data2)
len(sum_data4)


##  
data2_cust = sum_data2['고객번호']
data2_cust
final_data = original_data.loc[original_data['고객번호'].isin(data2_cust)]



