#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 17:15:24 2018

@author: philip
"""
import pandas as pd
data = pd.read_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/cluster_data.csv", sep=',')

#### 1. 주중, 주말로 구분하는 변수생성
data.info() # 이미 date 객체
data['구매일자'].head(10)
#data['구매일자'].dt.weekday_name # 요일을 문자열로 표현

# 주말여부 열 삽입
data['구매일자'].head(10)
# 구매일자를 datetime 타입으로 
data['구매일자'] = pd.to_datetime(data['구매일자'], errors='coerce')
# 요일을 숫자로 표현 : 0(월) ~ 6(일)
data['주말여부'] = data['구매일자'].dt.weekday 
# 확인
data[['구매일자','주말여부']].head(10)
data.info()




#### 2. 주방문시간대 분류 (6개 그룹)
data['구매시간그룹화'] = '0'
data['구매시간'].head(10)
#확인 
data['구매시간'].max()
data['구매시간'].min()
# 밤 12시 ~ 새벽 4시 : 1
# 새벽 4시 ~ 아침 8시 : 2
# 아침 8시 ~ 낮 12시 : 3
# 낮 12시 ~ 낮 4시 : 4
# 낮 4시 ~ 저녁 8시 : 5
# 저녁 8시 ~ 밤 24시 : 6
data.loc[(data["구매시간"] >= 0) & (data["구매시간"] < 4), "구매시간그룹화"] = '1'
data.loc[(data['구매시간'] >= 4) & (data['구매시간'] < 8), "구매시간그룹화"] = '2'
data.loc[(data['구매시간'] >= 8) & (data['구매시간'] < 12), "구매시간그룹화"] = '3'
data.loc[(data['구매시간'] >= 12) & (data['구매시간'] < 16), "구매시간그룹화"] = '4'
data.loc[(data['구매시간'] >= 16) & (data['구매시간'] < 20), "구매시간그룹화"] = '5'
data.loc[(data['구매시간'] >= 20) & (data['구매시간'] <= 23), "구매시간그룹화"] = '6'
# 확인
data[['구매시간', '구매시간그룹화']].head(10)
data.info()

### +주말여부 +구매시간그룹화 
data.to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/timegroup_data.csv", index=False)



