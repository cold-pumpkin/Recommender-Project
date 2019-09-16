#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 01:49:09 2018

@author: philip
"""

import pandas as pd

svd = pd.read_csv('/Users/philip/Workspace/Final_Project/svd.csv', sep=',', encoding='EUC-KR')
mf = pd.read_csv('/Users/philip/Workspace/Final_Project/matrix.csv', sep=',', encoding='EUC-KR')
xg = pd.read_csv('/Users/philip/Workspace/Final_Project/xgboost.csv', sep=',', encoding='EUC-KR')

svd.head()
mf.head()
xg.head()


### 누락된 고객 찾기
len(xg['고객번호'].unique())

xg_cust = list(xg['고객번호'].unique())
svd_cust = list(svd['고객번호'].unique())
mf_cust = list(mf['고객번호'].unique())

# matrix에서 누락된 고객
set(xg_cust) - set(mf_cust)     # {14599, 15999}

# svd에서 누락된 고객 
set(xg_cust) - set(svd_cust)    # {18761, 19209}


# 누락된 고객에 대해서는 가장 많이 구매한 소분류 삽입
purchase_info = pd.read_csv('/Users/philip/Workspace/Final_Project/Data/02.구매상품TR.csv', sep=',', engine='python')


# 가장 많이 구매한 소분류 top3
omitted_mf1 = purchase_info[purchase_info['고객번호']==14599][['고객번호', '소분류코드']]
omitted_mf1 = omitted_mf1.groupby(['고객번호', '소분류코드']).size().reset_index()
omitted_mf1.columns = ['고객번호', '소분류코드', '구매횟수']
omitted_mf1 = omitted_mf1.sort_values(['구매횟수'], ascending=[False]).reset_index(drop=True)
omitted_mf1 = omitted_mf1.head(3)
omitted_mf1 = omitted_mf1.drop('구매횟수', axis=1)
omitted_mf1


omitted_mf2 = purchase_info[purchase_info['고객번호']==15999][['고객번호', '소분류코드']]
omitted_mf2 = omitted_mf2.groupby(['고객번호', '소분류코드']).size().reset_index()
omitted_mf2.columns = ['고객번호', '소분류코드', '구매횟수']
omitted_mf2 = omitted_mf2.sort_values(['구매횟수'], ascending=[False]).reset_index(drop=True)
omitted_mf2 = omitted_mf2.head(3)
omitted_mf2 = omitted_mf2.drop('구매횟수', axis=1)
omitted_mf2

omitted_svd1 = purchase_info[purchase_info['고객번호']==18761][['고객번호', '소분류코드']]
omitted_svd1 = omitted_svd1.groupby(['고객번호', '소분류코드']).size().reset_index()
omitted_svd1.columns = ['고객번호', '소분류코드', '구매횟수']
omitted_svd1 = omitted_svd1.sort_values(['구매횟수'], ascending=[False]).reset_index(drop=True)
omitted_svd1 = omitted_svd1.head(3)
omitted_svd1 = omitted_svd1.drop('구매횟수', axis=1)
# 구매정보 1개

omitted_svd2 = purchase_info[purchase_info['고객번호']==19209][['고객번호', '소분류코드']]
omitted_svd2 = omitted_svd2.groupby(['고객번호', '소분류코드']).size().reset_index()
omitted_svd2.columns = ['고객번호', '소분류코드', '구매횟수']
omitted_svd2 = omitted_svd2.sort_values(['구매횟수'], ascending=[False]).reset_index(drop=True)
omitted_svd2 = omitted_svd2.head(3)
omitted_svd2 = omitted_svd2.drop('구매횟수', axis=1)
omitted_svd2
# 구매정보 1개 



# svd > mf > xg 순으로 item1 ~ item3
svd.head()
svd_item1 = svd[['고객번호', 'item']]
svd_item1.columns = ['고객번호', 'item1_svd']
omitted_svd1.columns = ['고객번호', 'item1_svd']
omitted_svd2.columns = ['고객번호', 'item1_svd']
svd_item1 = svd_item1.append(omitted_svd1)
svd_item1 = svd_item1.append(omitted_svd2)
svd_item1 = svd_item1.sort_values('고객번호').reset_index(drop=True)
svd_item1

svd_item2 = svd[['고객번호', 'item2']]
svd_item2.columns = ['고객번호', 'item2_svd']
omitted_svd1.columns = ['고객번호', 'item2_svd']
omitted_svd2.columns = ['고객번호', 'item2_svd']
svd_item2 = svd_item2.append(omitted_svd1)
svd_item2 = svd_item2.append(omitted_svd2)
svd_item2 = svd_item2.sort_values('고객번호').reset_index(drop=True)
svd_item2

svd_item3 = svd[['고객번호', 'item3']]
svd_item3.columns = ['고객번호', 'item3_svd']
omitted_svd1.columns = ['고객번호', 'item3_svd']
omitted_svd2.columns = ['고객번호', 'item3_svd']
svd_item3 = svd_item3.append(omitted_svd1)
svd_item3 = svd_item3.append(omitted_svd2)
svd_item3 = svd_item3.sort_values('고객번호').reset_index(drop=True)
svd_item3

mf_item1 = mf[['고객번호', 'item']]
mf_item1.columns = ['고객번호', 'item1_mf']
omitted_mf1_item1 = omitted_mf1.iloc[[0]]
omitted_mf2_item1 = omitted_mf2.iloc[[0]]
omitted_mf1_item1.columns = ['고객번호', 'item1_mf']
omitted_mf2_item1.columns = ['고객번호', 'item1_mf']
mf_item1 = mf_item1.append(omitted_mf1_item1)
mf_item1 = mf_item1.append(omitted_mf2_item1)
mf_item1 = mf_item1.sort_values('고객번호').reset_index(drop=True)
mf_item1

mf_item2 = mf[['고객번호', 'item2']]
mf_item2.columns = ['고객번호', 'item2_mf']
omitted_mf1_item2 = omitted_mf1.iloc[[1]]
omitted_mf2_item2 = omitted_mf2.iloc[[1]]
omitted_mf1_item2.columns = ['고객번호', 'item2_mf']
omitted_mf2_item2.columns = ['고객번호', 'item2_mf']
mf_item2 = mf_item2.append(omitted_mf1_item2)
mf_item2 = mf_item2.append(omitted_mf2_item2)
mf_item2 = mf_item2.sort_values('고객번호').reset_index(drop=True)
mf_item2

mf_item3 = mf[['고객번호', 'item3']]
mf_item3.columns = ['고객번호', 'item3_mf']
omitted_mf1_item3 = omitted_mf1.iloc[[2]]
omitted_mf2_item3 = omitted_mf2.iloc[[2]]
omitted_mf1_item3.columns = ['고객번호', 'item3_mf']
omitted_mf2_item3.columns = ['고객번호', 'item3_mf']
mf_item3 = mf_item3.append(omitted_mf1_item3)
mf_item3 = mf_item3.append(omitted_mf2_item3)
mf_item3 = mf_item3.sort_values('고객번호').reset_index(drop=True)
mf_item3

xg_item1 = xg[['고객번호', 'item']]
xg_item2 = xg[['고객번호', 'item2']]
xg_item3 = xg[['고객번호', 'item3']]
xg_item1.columns = ['고객번호', 'item1_xg']
xg_item2.columns = ['고객번호', 'item2_xg']
xg_item3.columns = ['고객번호', 'item3_xg']
xg_item1
xg_item2



top3 = pd.merge(svd_item1, mf_item1, on='고객번호')
top3 = pd.merge(top3, xg_item1, on='고객번호')
top3 = pd.merge(top3, svd_item2, on='고객번호')
top3 = pd.merge(top3, mf_item2, on='고객번호')
top3 = pd.merge(top3, xg_item2, on='고객번호')
top3 = pd.merge(top3, svd_item3, on='고객번호')
top3 = pd.merge(top3, mf_item3, on='고객번호')
top3 = pd.merge(top3, xg_item3, on='고객번호')
top3


## 정답 시트 만들기
answer = pd.DataFrame(columns=['고객번호', '추천상품1', '추천상품2', '추천상품3'])
answer['고객번호'] = 1
answer


top3[ top3.index==102]

top3
reco_arr = []

def insert_items(row):
    # 행 하나 당 실행 
    row_arr = []
    for idx in row.index:
        if(len(row_arr)==4):
            break
        # 고객번호 추가
        if idx=='고객번호':
            row_arr.append(row[idx])
        # 소분류코드 추가
        elif not pd.isnull(row[idx]) and row[idx][0].isalpha() and (row[idx] not in row_arr):
            row_arr.append(row[idx])
    reco_arr.append(row_arr)

top3.apply(insert_items, axis=1)
reco_df = pd.DataFrame(reco_arr)
reco_df.columns=['고객번호', '추천상품1', '추천상품2', '추천상품3']
reco_df.isnull().any()
reco_df.info()
reco_df



reco_df[reco_df['고객번호'].astype(str)==reco_df['추천상품1']]
reco_df[reco_df['고객번호'].astype(str)==reco_df['추천상품2']]
reco_df[reco_df['고객번호'].astype(str)==reco_df['추천상품3']]

reco_df[reco_df['추천상품2'].isnull()]
reco_df[reco_df['추천상품3'].isnull()]


# 상품분류  
cat_info = pd.read_csv('/Users/philip/Workspace/Final_Project/Data/03.상품분류.csv', sep=',', engine='python')
cat_name = cat_info[['소분류코드', '소분류명']]

# 고객정보 
cust_info = pd.read_csv('/Users/philip/Workspace/Final_Project/Data/01.고객DEMO.csv', sep=',', engine='python')
cust_info

# +상품분류 병합
reco_id = reco_df['고객번호']
reco_id = pd.DataFrame(reco_id)
reco_id.columns = ['고객번호']
reco_id

reco1 = reco_df['추천상품1']
reco1 = pd.DataFrame(reco1)
reco1.columns = ['소분류코드']
reco1 = pd.merge(reco1, cat_name, on='소분류코드', how='left')
reco1

reco2 = reco_df['추천상품2']
reco2 = pd.DataFrame(reco2)
reco2.columns = ['소분류코드']
reco2 = pd.merge(reco2, cat_name, on='소분류코드', how='left')
reco2

reco3 = reco_df['추천상품3']
reco3 = pd.DataFrame(reco3)
reco3.columns = ['소분류코드']
reco3 = pd.merge(reco3, cat_name, on='소분류코드', how='left')
reco3

reco_final = pd.concat([reco_id, reco1, reco2, reco3], axis=1)
reco_final.isnull().any()

# +고객정보 병합
final_reco = pd.concat([reco_final, cust_info.iloc[:,1:]], axis=1)
final_reco.to_csv('/Users/philip/Workspace/Final_Project/Data/final_reco.csv', index=False)

final_reco.to_excel('/Users/philip/Workspace/Final_Project/Data/final_reco.xlsx', sheet_name='최종추천')
final_reco[final_reco['고객번호']==19209]
