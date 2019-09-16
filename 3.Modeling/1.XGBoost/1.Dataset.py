#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 14:18:49 2018

@author: philip
"""
import pandas as pd 

data = pd.read_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/classification_data.csv", sep=',')
data.head(30)
# 결측치 확인
data.isnull().values.any() 
# 열별 결측치 개수 확인
data.isnull().sum() 

### 1. 거주지역이 NaN이면 0으로 처리
#   -> ok

### 2. 멤버십이 NaN이면 0으로 처리
data.info()
data[['하이마트보유', '다둥이보유', '롭스보유', '더영보유']] = data[['하이마트보유', '다둥이보유', '롭스보유', '더영보유']].fillna(0)
data[['하이마트보유', '다둥이보유', '롭스보유', '더영보유']] = data[['하이마트보유', '다둥이보유', '롭스보유', '더영보유']].astype(int)
data[['하이마트보유', '다둥이보유', '롭스보유', '더영보유']]

### 3. 채널 이용횟수
data = data.rename(columns={'A_MOBILE/APP': 'A모바일/앱', 'B_MOBILE/APP': 'B모바일/앱',
                            'C_MOBILE/APP': 'C모바일/앱', 'D_MOBILE/APP': 'D모바일/앱',
                            'B_ONLINEMALL': 'B온라인몰',  'C_ONLINEMALL': 'C온라인몰'})

data.info()

data.to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/classification_data.csv", index=False)
