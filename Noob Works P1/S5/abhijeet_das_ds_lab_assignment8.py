# importing libs

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

# importing data from URL

url = 'https://raw.githubusercontent.com/akaakselabhijeet/Data-Science-Datasets-Ver.01/main/India_Data/India%20Election%20Dataset%201.csv'
ls = pd.read_csv(url)
print(ls.head(5))
ls.info()

# make a clone

ls1 = ls.copy()
print(ls1.head(4))

# Applying discretizer on GenVotes

discretizer = KBinsDiscretizer()
discretizer = discretizer.fit_transform(ls1['GenVotes'].values.reshape(-1,1))
print(discretizer)

# To_Array

disc1 = pd.DataFrame(discretizer.toarray(), columns=['binary_'+str(i) for i in range(1, 6)])
print(disc1.head(6))
disc1.info()

# Implementing Binarizer

binarizer = Binarizer(threshold = 30000)
binarizer = binarizer.fit_transform(ls1['TotalVotes'].values.reshape(-1,1))
print(binarizer)

# new_dataframe

bin2 = pd.DataFrame(binarizer, columns=['TotalVotesBin'])
print(bin2.head(4))
print(bin2.tail(4))

# concat function on disc1, bin2, ls1

ls2 = pd.concat([disc1, bin2, ls1], axis=1)
print(ls2.head(6))

# Dataframe without 'Winner'

a1 = ls2.drop('Winner', axis=1)
print(a1.head(6))

# Dataframe with 'Winner'

a2 = pd.DataFrame(ls2['Winner'])
print(a2.tail(6))

# Splitting the data in a1 and a2 in training and test data using train_test_split

X_train, X_test, y_train, y_test = train_test_split(a1, a2, test_size=0.15)
print("operation successful!")

# Implementing Standardization with RobustScaler

rscaler = RobustScaler(quantile_range=(0.25, 0.75))
rscaler = rscaler.fit_transform(a1['Age'].values.reshape(-1, 1))
print(rscaler)

# result data into dataframe scaler1

scaler1 = pd.DataFrame(rscaler, columns=['Scaled_Age'])
print(scaler1.head(6))

# a2 copy with selected clone

a2 = a1[['GenVotes', 'PostalVotes', 'TotalVotes']].copy()
print(a2.head(6))

# Implementing Normalization with Normalizer(norm='max') on a2

normalizer = Normalizer(norm='max')
normalizer = normalizer.fit_transform(a2)
print(normalizer)

# result data in dataframe 'normal2'

normal2 = pd.DataFrame(normalizer, columns=a2.columns)
print(normal2.head(8))

# THE END
# Abhijeet Das - DS Assignment 8