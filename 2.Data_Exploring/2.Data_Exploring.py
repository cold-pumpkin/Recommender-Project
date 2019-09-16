#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:33:12 2018

@author: philip
"""
import pandas as pd
import numpy as np

#merged_Data = pd.read_csv("/Users/philip/Workspace/Final_Project/R_to_Python/merged_Data.csv", sep=',', encoding = "EUC-KR")
classified_data = pd.read_csv("/Users/philip/Workspace/Final_Project/R_to_Python/classified_data.csv", sep=',', encoding = "EUC-KR")


classified_data.drop(['영수증번호', '구매시간', '가입년월', '하이마트보유', '다둥이보유', '롭스보유', '더영보유',
                      '구매금액', 'A_MOBILE/APP', 'B_MOBILE/APP', 'C_MOBILE/APP',
                      'D_MOBILE/APP','B_ONLINEMALL', 'C_ONLINEMALL'], axis=1)

# 결측치 확인
classified_data.isnull().values.any() 
# 열별 결측치 개수 확인
classified_data.isnull().sum() 
