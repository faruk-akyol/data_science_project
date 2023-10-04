# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 22:54:15 2023

@author: faruk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model import train_test_split


df = pd.read_csv("data_eda.csv")


# relevant columns

df_relevant = df[["avg_salary","Rating", "Size", "Type of ownership",
                  "Industry","Sector","Revenue","comp_num",
                  "hourly","employer_pro","job_state","hq_state","hq_city","job_city",
                  "same_state","same_city","age","python_des","aws_des",
                  "spark_des","azure_des","hadoop_des","scikit-learn_des",
                  "keras_des","google cloud_des","seniority","job_simp","desc_len"]]

# get dummy data

df_dum = pd.get_dummies(df_relevant)
# train-test split

# multiple linear regression

# lasso regression 

# random forest





