#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:09:42 2018

@author: philip
"""

import pandas as pd
lifecycle_A = pd.read_csv("/Users/philip/Workspace/Final_Project/lifecycle/lifecycle_A.csv", sep=',')
lifecycle_B = pd.read_csv("/Users/philip/Workspace/Final_Project/lifecycle/lifecycle_B.csv", sep=',')
lifecycle_C = pd.read_csv("/Users/philip/Workspace/Final_Project/lifecycle/lifecycle_C.csv", sep=',')
lifecycle_D = pd.read_csv("/Users/philip/Workspace/Final_Project/lifecycle/lifecycle_D.csv", sep=',')

lifecycle_all = pd.concat([lifecycle_A, lifecycle_B, lifecycle_C, lifecycle_D]) 
lifecycle_all.columns
del lifecycle_all['Unnamed: 0']
lifecycle_all

lifecycle_all.to_csv("/Users/philip/Workspace/Final_Project/lifecycle/lifecycle_all.csv")



### 
