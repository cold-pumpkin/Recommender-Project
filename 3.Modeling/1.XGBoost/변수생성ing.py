#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:34:55 2018

@author: philip
"""
import pandas as pd
import numpy as np
pd.set_option('display.expand_frame_repr', False) #모든 열 보이게 하기

#%%
########################################
############### (구매정보 + 고객정보) 데이터 로드
########################################
buying_cust = pd.read_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/buying+cust.csv", sep=',')
buying_cust.info()
buying_cust.head()



#%%
########################################
############### (구매정보 + 고객정보 + 상품분류) 데이터 merge
########################################
product_info = pd.read_csv("/Users/philip/Workspace/Final_Project/Data/03.상품분류.txt", sep=',', encoding = "EUC-KR")
product_info.info()
# merge 전에 겹치는 열 제거
product_info = product_info.drop(['제휴사', '대분류코드', '중분류코드'], axis=1)
product_info.info()
buying_cust.info()

buying_prod = pd.merge(buying_cust, product_info, on='소분류코드')
buying_prod.info()
buying_prod.head(10)



#%%
########################################
############### (구매정보 + 고객정보 + 상품분류 + 자주가는제휴사) 
############### 자주가는 제휴사 구하기 : 최근 30 % 
########################################


from datetime import datetime


### 기존 데이터셋 정렬

# 구매일자 : int -> str -> date 타입 변환
buying_prod['구매일자'] = buying_prod['구매일자'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))

# 고객번호(오름차순), 구매일자(내림차순) 기준으로 정렬
buying_prod = buying_prod.sort_values(['고객번호', '구매일자'], ascending=[True, False])
buying_prod.sort_values(['고객번호', '구매일자', '영수증번호'], ascending=[True, False, True])

# 인덱스 리셋 (+ 인덱스 열 제거)
buying_prod = buying_prod.reset_index(drop=True)
buying_prod.head(10)


#buying_prod.sort_values(['고객번호', '구매일자'], ascending=[True, False])



#%%

### 고객/구매일자/제휴사별로 구매품목 수 구하기

buying_prod.head(20)

# 고객별/구매일자/제휴사를 그룹화하여 데이터 카운트 -> '구매품목수'
count_company = buying_cust.groupby(['고객번호', '구매일자', '제휴사']).size().reset_index(name='구매품목수')
count_company.head(10)

# 고객번호(오름차순), 구매일자(내림차순) 기준으로 정렬
count_company = count_company.sort_values(['고객번호', '구매일자'], ascending=[True, False])
count_company = count_company.reset_index(drop=True)
count_company
# : 고객번호, 구매일자, 제휴사, 구매품목수




#%%

### 자주가는 계열사 구하기 
###    : 고객별로 최근 1/2 기간안에 가장 자주 이용한 제휴사

count_company.head(10)
count_company[count_company['고객번호']==1]
# 고객별 방문횟수
count_company.groupby('고객번호').size() 
count_company

count_company1 = pd.read_csv("/Users/philip/Workspace/count_company.csv", sep=',', encoding='EUC-KR')
count_company1.columns = ['고객번호', '구매일자', '제휴사', '구매품목수']
count_company1




diff = count_company['제휴사'] == count_company1['제휴사']
diff
count_company['고객번호'] 

cc = pd.DataFrame(count_company['고객번호'] )
cc
diff = pd.DataFrame(diff)
diff

diff = pd.concat([diff.reset_index(drop=True), count_company['고객번호']], axis=1)
diff[diff['제휴사']==False]
diff[diff['제휴사']==True]


count_company.equals(count_company1)



#count_company_r = pd.read_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/count_company.csv", sep=',')
#count_company.equals(count_company_r)
#count_company_r.head(200)
#count_company.to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/count_company_py.csv", index=False)

# 최근 1/2 방문 중 가장 많이 방문한 제휴사 구하는 함수
def frequent_company(x):
    y = (x.size/2)
    y = int(y)
    return x.iloc[:y, 2].value_counts().idxmax() 

# 고객번호별 그룹화하여 자주가는 제휴사 구하기

count_company
count_company.groupby('고객번호').size()

freq_company = count_company.groupby('고객번호').apply(frequent_company)
freq_company = pd.DataFrame(freq_company, columns=['자주가는제휴사']).reset_index()
freq_company
# 결측치 확인
#freq_company.isnull().values.any() 

#type(freq_company['자주가는제휴사'])
#freq_company['자주가는제휴사'].equals(result['자주가는제휴사'])

#%%

### 확인
len(freq_company[freq_company['자주가는제휴사']=='A']) 
len(freq_company[freq_company['자주가는제휴사']=='B']) 
len(freq_company[freq_company['자주가는제휴사']=='C'])  
len(freq_company[freq_company['자주가는제휴사']=='D'])


#%%

### 데이터 merge : 구매정보 + 고객정보 + 자주가는제휴사
## 종은's result
freq_company1 = pd.read_csv("/Users/philip/Workspace/result_id.csv", sep=',', encoding='EUC-KR')
freq_company1
len(freq_company1[freq_company1['자주가는제휴사']=='A']) 
len(freq_company1[freq_company1['자주가는제휴사']=='B']) 
len(freq_company1[freq_company1['자주가는제휴사']=='C'])  
len(freq_company1[freq_company1['자주가는제휴사']=='D'])

data = pd.merge(buying_prod, freq_company1, on='고객번호')
data.info()
data.to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/freq_company.csv", index=False)



#%%
########################################
############### (구매정보 + 고객정보 + 상품분류 + 자주가는제휴사 + 멤버십보유) 
########################################

### 멤버십 보유 여부 확인하여 merge
import pandas as pd

# (구매정보 + 고객정보 + 상품분류 + 자주가는제휴사) 데이터 로드 
data = pd.read_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/freq_company.csv", sep=',')
membership_info = pd.read_csv("/Users/philip/Workspace/Final_Project/Data/05.멤버십여부.txt", sep=',', encoding = "EUC-KR")
membership_info.isnull().values.any()


# 멤버십명 데이터 확인
membership_info['멤버십명'].unique()

# 가입년월 열 타입을 모두 int -> string 타입으로 바꾸기
membership_info['가입년월'] = membership_info['가입년월'].astype(str)
membership_info


# 고객번호로 그룹화하여 고객들의 멤버십 가입 개수 확인
a = membership_info.groupby(['고객번호']).agg({'고객번호':'count'})
a.columns = ['가입개수']
a = a.reset_index()

# 가입 개수 확인
len(a[a['가입개수']==1]) # 5601
len(a[a['가입개수']==2]) # 837
len(a[a['가입개수']==3]) # 59
len(a[a['가입개수']==4]) # 1

# 고객번호로 그룹화하여 멤버십명 한 행으로 합치기
membership_info.groupby('고객번호')['멤버십명'].apply(', '.join)
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
#import numpy as np

membership_tuning['하이마트보유'] = np.where(membership_tuning['멤버십명'].str.contains('하이마트'), '1', '0')
membership_tuning['다둥이보유'] = np.where(membership_tuning['멤버십명'].str.contains('다둥이'), '1', '0')
membership_tuning['롭스보유'] = np.where(membership_tuning['멤버십명'].str.contains('롭스'), '1', '0')
membership_tuning['더영보유'] = np.where(membership_tuning['멤버십명'].str.contains('더영'), '1', '0')


membership = membership_tuning[['고객번호', '하이마트보유', '다둥이보유', '롭스보유', '더영보유']]



# (고객정보 + 구매정보 + 상품분류 + 자주가는제휴사) 파일과 멤버십 파일을 merge
merged_member = pd.merge(data, membership, on='고객번호', how='left')

merged_member.head()
merged_member.info()
merged_member.head(10)
merged_member = merged_member.fillna('0')

len(merged_member[merged_member['자주가는제휴사']=='A'])
len(merged_member[merged_member['자주가는제휴사']=='B'])
len(merged_member[merged_member['자주가는제휴사']=='C'])
len(merged_member[merged_member['자주가는제휴사']=='D'])

merged_member.to_csv("/Users/philip/Workspace/Final_Project/R_to_Python/56789/dataset/merged_membership.csv", index=False)



#%%
