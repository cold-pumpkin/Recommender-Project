#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 17:52:49 2018

"""
#%%
########### 1. 원본 데이터 로드 ###########

## 데이터 로드의 편의를 위해 원본 데이터셋의 이름을 다음과 같이 변경
# 01.고객DEMO.txt   02.구매상품TR.txt
# 03.상품분류.txt    04.경쟁사이용.txt
# 05.멤버십여부.txt   06.채널이용.txt

## 데이터 전처리 라이브러리 import
import pandas as pd
import numpy as np
pd.set_option('display.expand_frame_repr', False) #모든 열 보이게 하기

#%%
## 1) 고객정보 데이터 읽기
cust_info = pd.read_csv("/Users/philip/Workspace/Final_Project/Data/01.고객DEMO.txt", sep=',', encoding = "EUC-KR")
cust_info.info()
# NaN -> 0
cust_info = cust_info.fillna(0) 
# 열 정수화 
cust_info['거주지역'] = cust_info['거주지역'].astype(int)
cust_info.head(10)


## 2) 구매상품 데이터 읽기
buying_info = pd.read_csv("/Users/philip/Workspace/Final_Project/Data/02.구매상품TR.txt", sep=',', encoding = "EUC-KR")
buying_info.isnull().values.any() 
buying_info.head(10)


## 3) 상품분류 데이터 읽기
product_info = pd.read_csv("/Users/philip/Workspace/Final_Project/Data/03.상품분류.txt", sep=',', encoding = "EUC-KR")
product_info.isnull().values.any() 
product_info.head(10)


## 4) 경쟁사 이용 데이터 읽기
competitor_info = pd.read_csv("/Users/philip/Workspace/Final_Project/Data/04.경쟁사이용.txt", sep=',', encoding = "EUC-KR")
competitor_info.isnull().values.any() 
competitor_info.head(10)


## 5) 멤버십 여부 데이터 읽기 
membership_info = pd.read_csv("/Users/philip/Workspace/Final_Project/Data/05.멤버십여부.txt", sep=',', encoding = "EUC-KR")
membership_info.isnull().values.any()
membership_info.head(10)


## 6) 채널이용 데이터 읽기
channel_info = pd.read_csv("/Users/philip/Workspace/Final_Project/Data/06.채널이용.txt", sep=',', encoding = "EUC-KR")
channel_info.isnull().values.any()
channel_info.head(10)

#%%

## csv 파일 생성
cust_info.to_csv("/Users/philip/Workspace/Final_Project/Data/01.고객DEMO.csv", index=False)
buying_info.to_csv("/Users/philip/Workspace/Final_Project/Data/02.구매상품TR.csv", index=False)
product_info.to_csv("/Users/philip/Workspace/Final_Project/Data/03.상품분류.csv", index=False)
competitor_info.to_csv("/Users/philip/Workspace/Final_Project/Data/04.경쟁사이용.csv", index=False)
membership_info.to_csv("/Users/philip/Workspace/Final_Project/Data/05.멤버십여부.csv", index=False)
channel_info.to_csv("/Users/philip/Workspace/Final_Project/Data/06.채널이용.csv", index=False)

#%%

############ 2. 구매상품 x 고객정보 merge : inner join ###########

buying_info.info()
cust_info.info()
merged_buying = pd.merge(buying_info, cust_info, on="고객번호", how="inner")

# 고객번호 기준으로 정렬
merged_buying = merged_buying.sort_values(['고객번호'])

# 열 순서 맞추기
merged_buying = merged_buying[['고객번호', '제휴사', '영수증번호', '대분류코드', '중분류코드', '소분류코드', '점포코드', '구매일자', '구매시간', '구매금액', '성별', '연령대', '거주지역']]
merged_buying = merged_buying.reset_index(drop=True)

# 확인
merged_buying.info()
merged_buying.head(10)
merged_buying.tail(10)

# csv파일로 저장
merged_buying.to_csv("/Users/philip/Workspace/Final_Project/Data/buying+cust.csv", index=False)
merged_buying.info()

############################
############################
############################
############################
############################
############################
####### 여기까지 merged ######
############################
############################
############################
############################
############################
############################

#%%




###### 멤버십 & 자주가는 제휴사 

############ 3. membership_info 변수 가공 : 중복아이디 병합 및 피벗팅, 멤버십 테이블 Merge ###########





# 멤버십명 데이터 확인
membership_info['멤버십명'].unique()
# 하이마트 : 
# 다둥이 : 
# 롭스 :  
# 더영 : 

# 가입년월 열 타입을 모두 int -> string 타입으로 바꾸기
membership_info['가입년월'] = membership_info['가입년월'].astype(str)
membership_info


# 고객번호로 그룹화하여 고객들의 멤버십 가입 개수 확인
a = membership_info.groupby(['고객번호']).agg({'고객번호':'count'})
a.columns = ['가입개수']
a = a.reset_index()



len(a[a['가입개수']==1]) # 5601
len(a[a['가입개수']==2]) # 837
len(a[a['가입개수']==3]) # 59
len(a[a['가입개수']==4]) # 1

# 고객번호로 그룹화하여 멤버십명 한 행으로 합치기
joined_membership1 = membership_info.groupby('고객번호')['멤버십명'].apply(', '.join)


# 고객번호로 그룹화하여 가입년월 한 행으로 합치기
joined_membership2 = membership_info.groupby('고객번호')['가입년월'].apply(', '.join)
joined_membership1 = pd.DataFrame(joined_membership1)
joined_membership2 = pd.DataFrame(joined_membership2)
joined_membership1.columns = ['멤버십명']
joined_membership1.reset_index()
joined_membership2.columns = ['가입년월']
joined_membership2.reset_index()


# merge
membership_tuning = pd.merge(joined_membership1, joined_membership2, on=['고객번호']) 
membership_tuning = membership_tuning.reset_index()


# 멤버십가입 여부를 1/0으로 표현
membership_tuning['하이마트보유'] = np.where(membership_tuning['멤버십명'].str.contains('하이마트'), '1', '0')
membership_tuning['다둥이보유'] = np.where(membership_tuning['멤버십명'].str.contains('다둥이'), '1', '0')
membership_tuning['롭스보유'] = np.where(membership_tuning['멤버십명'].str.contains('롭스'), '1', '0')
membership_tuning['더영보유'] = np.where(membership_tuning['멤버십명'].str.contains('더영'), '1', '0')
membership_tuning


# 고객정보+구매이력 파일과 멤버십 파일을 merge
merged_member = pd.merge(merged_buying, membership_tuning, on='고객번호', how='left')
merged_member.info()
merged_member.tail(10)

#merged_member.to_csv("/Users/philip/Workspace/Final_Project/Data/.csv", index=False)
#%%

############ 4. 채널 정보 Merge ###########
channel_info
channel_pivot = channel_info.pivot_table(index=['고객번호'], columns='제휴사', values='이용횟수').reset_index()
channel_pivot = channel_pivot.fillna(0)
channel_pivot = channel_pivot.astype(int)
channel_pivot

merged_channel = pd.merge(merged_member, channel_pivot, on='고객번호', how='left')
merged_channel.to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/merged_Data.csv", index=False)


############ 5. 기본 데이터셋 구성 ###########
merged_channel.info()
product_info.info()

merged_channel = merged_channel.drop(['멤버십명', '가입년월'], axis=1)
product_info = product_info[['중분류명','소분류명','소분류코드']]

#상품 정보 Merge
join_prod_name = pd.merge(merged_channel, product_info, on='소분류코드', how='left')
join_prod_name.info()
join_prod_name.tail(100)

join_prod_name[['A_MOBILE/APP',  'B_MOBILE/APP',  'B_ONLINEMALL',  'C_MOBILE/APP',  'C_ONLINEMALL', 'D_MOBILE/APP']] = join_prod_name[['A_MOBILE/APP',  'B_MOBILE/APP',  'B_ONLINEMALL',  'C_MOBILE/APP',  'C_ONLINEMALL', 'D_MOBILE/APP']].fillna(0)
join_prod_name[['A_MOBILE/APP',  'B_MOBILE/APP',  'B_ONLINEMALL',  'C_MOBILE/APP',  'C_ONLINEMALL', 'D_MOBILE/APP']] = join_prod_name[['A_MOBILE/APP',  'B_MOBILE/APP',  'B_ONLINEMALL',  'C_MOBILE/APP',  'C_ONLINEMALL', 'D_MOBILE/APP']].astype(int)

join_prod_name.head(10)
join_prod_name.to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/classification_data.csv", index=False)
