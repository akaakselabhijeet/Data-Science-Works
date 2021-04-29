# ABHIJEET DAS
# DATA SCIENCE ASSIGNMENT 7
# APRIL 29, 2021

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# import dataset
# for dataset, check my repo

url = "https://raw.githubusercontent.com/akaakselabhijeet/Data-Science-Datasets-Ver.01/main/India_Data/India%20Election%20Dataset%201.csv"
ls = pd.read_csv(url)
print(ls.head(5))

#Cloning dataframe

ls1 = ls.copy()
ls2 = ls.copy()
ls3 = ls.copy()

print(ls3.head(5))

#Drop null valued columns

ls1 = ls1.dropna(axis=1)
print(ls1.head(5))

#Drop null valued rows

ls2 = ls2.dropna(axis=0)
print(ls2.head(5))

# iterating over the columns

for col in ls3.columns:
    print(col)

#finding mean values

ls4 = ls
mn = ls4['Age'].mean()
print("Mean Age Value: ",mn)
print("\n")

# fill NaN values with column mean

ls3['Age'].fillna(value=mn, inplace=True)
print(ls3['Age'].head(5))

#cloning with selected cols

ls5 = ls[['Age','TotalVotes','PostalVotes']].copy()
print(ls5.head(6))

#Checking if there is any null NaN values

print(ls5[ls5.isna().any(axis=1)])
print(ls5.isna())

#SimpleImputer module usage:
#Fill all the null Age values with mean of Age in ls5 using SimpleImputer module.

imputer = SimpleImputer(strategy ='mean')
imputer = imputer.fit(ls5[['Age']])
ls5['Age'] = imputer.transform(ls5[['Age']])

print(ls5[['Age']].head(5))
print(ls5.head(5))

#Checking again if there is any null NaN values

print(ls5[ls5.isna().any(axis=1)])
print(ls5.isna())

#polynomial feature set from ls5 with degree 3. Concatenate the feature set
#with ls5 and create a new data frame ls6.

polynm = PolynomialFeatures(degree=3)
ls6 = pd.DataFrame(polynm.fit_transform(ls5))
print(ls6.head(5))
ls6 = pd.concat([ls5,ls6])
print(ls6.head(5))

# A new copy of main dataframe

ls7 = ls.copy()
print(ls7.head(8))

#Checking again if there is any null NaN values

ls8 = ls7[['Age']]

print(ls8[ls8.isna().any(axis=1)])
print(ls8.isna())
print(ls8.head(5))

# Replace all the null values in Gender and Category columns using the default value
#‘Missing’.

imputer = SimpleImputer(strategy='constant', fill_value='MISSING')
imputer2 = SimpleImputer(strategy='constant', fill_value=0)

ls7['Gender'] = imputer.fit_transform(ls7[['Gender']])
ls7['Category'] = imputer.fit_transform(ls7[['Category']])
ls7['Age'] = imputer2.fit_transform(ls7[['Age']])

#Checking again if there is any null NaN values

ls8 = ls7[['Age']]

print(ls8[ls8.isna().any(axis=1)])
print(ls8.isna())
print(ls8.head(5))

#Again comparing values

print(ls7[['Gender']].head(8))
print(ls[['Gender']].head(8))

# Creating ls13 for later concat use

ls13 = ls7.copy()

# Ordinal Encoder:

enco = OrdinalEncoder()
enco.fit(ls7[["Gender"]])
ls7[["Gender"]] = enco.transform(ls7[["Gender"]])
print("operation successful...")

#Comparing different outputs

print(ls7[['Gender']].head(8))
print(ls[['Gender']].head(8))

# checking the category column

print(ls7[['Category']].head(10))
print(ls7[['Gender']].head(10))

# Applying One-Hot_Encoder

ohe = OneHotEncoder()

ohe_df = pd.DataFrame(ohe.fit_transform(ls7[['Category']]).toarray())
print(ohe_df.head(15))

#Concat Operation (including major outputs)

ls10 = ls[['Gender']] # original column
ls13 = ls13[['Gender']] # imputed(missing-tagged) column
ls11 = ls7[['Category']] # imputed category column
horizontal_stack = pd.concat([ls10, ls13, ls11, ohe_df], axis=1)

print(horizontal_stack.head(21))

# THE END
# THANK YOU