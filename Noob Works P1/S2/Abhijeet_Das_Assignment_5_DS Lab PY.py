# -*- coding: utf-8 -*-
"""Abhijeet Das - Assignment 5 - DS.ipynb

'''
Abhijeet Das/A5/DS
DT: 12.04.2021
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Import dataset

url = "https://raw.githubusercontent.com/akaakselabhijeet/Data-Science-Datasets-Ver.01/main/India_Data/India%20Election%20Dataset%201.csv"
df = pd.read_csv(url)

#Verify it

print(df.head(5))

#Histogram for 'TotalVotes' Col

df['TotalVotes'].plot.hist()

#Scatter plot: 'Age' vs 'TotalVotes'

df.plot.scatter(x='Age', y='TotalVotes', title='Age vs TotalVotes')

#Bar plot for category column sorted by its values

df['Category'].value_counts().sort_index().plot.bar()

#Simple square plots

x = np.arange(1, 7)
print(x)

plt.plot(x, x**2)
plt.show()

plt.scatter(x, x**2)

#THE END

