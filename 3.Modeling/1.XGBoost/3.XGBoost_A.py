#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 08:41:45 2018

@author: philip
"""

import pandas as pd
pd.set_option('display.expand_frame_repr', False) #모든 열 보이게 하기


########################################
########### XGBoost : 제휴사 A ###########
########################################

 
### A제휴사를 최근에 자주 방문한 고객은 A제휴사의 상품 내에서 상품 추천(대분류코드)


###1. 학습용 데이터 생성
data = pd.read_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/final_data.csv", sep=',')
data.info()
data['새대분류코드'].head(5)

#활용하지 않는 변수 제거
#xg_data = data.drop(['새대분류코드','소분류코드', '구매일', '주말여부', '구매시간그룹화', 
#                     '제휴사', '영수증번호', '중분류코드', '구매시간', 
#                     '구매금액', '새중분류코드', '대분류코드', '자주가는제휴사'], axis=1)
# ['주말여부' '구매시간그룹화' '자주가는제휴사'] -> 왜 안들어갔지??
#

### XGBoost 타입에 맞게 타입변환(문자=>숫자)
# 현재 열 확인
'''
'고객번호', '점포코드', '구매일자', '성별', '연령대', '거주지역', '하이마트보유', '다둥이보유',
'롭스보유', '더영보유', 'A모바일/앱', 'B모바일/앱', 'B온라인몰', 'C모바일/앱', 'C온라인몰',
'D모바일/앱', '중분류명', '소분류명', '고객클러스터', '구매상품종류수', '구매년도', '구매월'
'''

### 2. Testset 생성 및 타입변환 
# 자주가는 제휴사별로 나눈 Test set 
test = pd.read_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/자주가는제휴사/A_testset.csv", sep=',', encoding = "EUC-KR")
test.info()
test = test.drop(['자주가는제휴사'], axis=1)
test_gender = test['성별']

data = data.rename(columns={'A모바일앱': 'A모바일/앱', 'B모바일/앱': 'B모바일/앱',
                            'C모바일앱': 'C모바일/앱', 'D모바일/앱': 'D모바일/앱'})


#pd.to_numeric(test_gender)


test_gender
test_age = test['연령대']
test_age



# XGBoost : 모든 타입이 숫자
test['성별'] = pd.factorize(test['성별'])[0]
test['연령대'] = pd.factorize(test['연령대'])[0]
test['중분류명'] = pd.factorize(test['중분류명'])[0]
test['소분류명'] = pd.factorize(test['소분류명'])[0]

# test 타입 변환
test.info()
test['성별']
test['연령대']
test['중분류명'].unique()
test['소분류명'].unique()


# xg_data 타입변환


### XGBoost
