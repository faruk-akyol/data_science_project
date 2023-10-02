# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:36:30 2023

@author: faruk
"""

import pandas as pd 
import numpy as np



df =pd.read_csv("glassdoor_jobs.csv")



df = df[df["Salary Estimate"] != "-1"]




df["hourly"] = df["Salary Estimate"].apply(lambda x: 1 if "per hour" in x.lower() else 0)
df["employer_pro"] = df["Salary Estimate"].apply( lambda x : 1 if "employer provided" in x.lower() else 0)


df["Salary"] = df["Salary Estimate"].apply(lambda x : x.lower().replace("(glassdoor est.)" , "").replace("per hour",""))
df["Salary"] = df["Salary"].apply(lambda x: x.lower().replace("(employer est.)" , "").replace("employer provided salary:",""))
df["Salary"] = df["Salary"].apply( lambda x : x.lower().replace("$", "").replace("k",""))
df["min_salary"] = df["Salary"].apply( lambda x :   int(x.split("-")[0]) )
df["max_salary"] = df["Salary"].apply( lambda x :   int(x.split("-")[1]) )



df["avg_salary"] = (df["min_salary"] + df["max_salary"]) / 2


#company name only
df["company_txt"] = df.apply(lambda x: -1 if x["Rating"] < 0 else x["Company Name"][:-3], axis=1)


# job location and headquarters of companies parsing
# binray table for same_state and same_city for headquarters and job location

df["job_city"] = df["Location"].apply(lambda x : x.split(", ")[0] )
df["job_state"] = df["Location"].apply(lambda x : x.split(", ")[1] )

headquarters_parsing = df["Headquarters"].apply(lambda x : x.split(",") )

df["hq_city"] = df["Headquarters"].apply(lambda x : "-1" if x=="-1" else x.split(", ")[0] )
df["hq_state"] = df["Headquarters"].apply(lambda x : "-1" if x=="-1" else x.split(", ")[1] )

df["same_city"] = np.where((df["hq_state"] != "-1") & (df["hq_city"] == df["job_city"]) & (df["hq_state"] == df["job_state"]), 1, 0)

df["same_state"] = np.where((df["hq_state"] != "-1")  & (df["hq_state"] == df["job_state"]), 1, 0)


#age of company

df["age"] = df["Founded"].apply(lambda x : 2023-x if x>0 else -1)

#parsing job description



# python , aws , spark , azure , hadoop ,Scikit-learn, PyTorch, TensorFlow and Keras, Google Cloud
wanted_suff = [ "python" , "aws" , "spark" , "azure" , "hadoop" ,"scikit-learn", "pytorch", "tensorflow", "keras", "google cloud"]
for elements in wanted_suff:
    df[elements+"_des"] = df["Job Description"].apply(lambda x : 1 if elements in x.lower() else 0)

df=df.drop("Unnamed: 0", axis=1)


df.to_csv("salary_data_cleaned.csv")










