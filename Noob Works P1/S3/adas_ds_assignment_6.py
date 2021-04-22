#ABHIJEET DAS
#ASSIGNMENT 6

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

url2 = "https://raw.githubusercontent.com/akaakselabhijeet/Data-Science-Datasets-Ver.01/main/Transport%20Data/Passenger%20data.csv"
url1 = "https://raw.githubusercontent.com/akaakselabhijeet/Data-Science-Datasets-Ver.01/main/India_Data/India%20Election%20Dataset%201.csv"
df1 = pd.read_csv(url1)
df2 = pd.read_csv(url2)
print(df1.head(5))
print(df2.head(5))

#Draw a scatter plot: Age vs. TotalVotes. hue = 'Education'

sns.scatterplot(x='Age', y='TotalVotes', hue='Education', data=df1)
plt.show()

#Draw a scatter plot: Age vs. TotalVotes. hue = 'Education' style = 'Gender'

sns.scatterplot(x='Age', y='TotalVotes', hue='Education', style='Gender', data=df1)
plt.show()

#Histogram for 'TotalVotes' Col

df1['TotalVotes'].plot.hist()

#Box plot: Age vs. Category And hue should be on Gender.

sns.boxplot(x='Age', y='Category', hue='Gender', data=df1)
plt.show()

#Select only numeric data in entire dataframe

df3 = df1._get_numeric_data()
df3.head(5)

#Heatmap with df3

sns.heatmap(data=df3)
plt.show

#Working with Dataset 2:

print(df2.head(5))

#Shape of dataset and its columns

print(df2.shape)

print(list(df2.columns))

#Print null values in each column

print(df2.isnull().sum())

#who survived, using pivot table

out = df2.pivot_table(index=['Gender'], columns=['Pclass'], aggfunc='count')
print(out)

#Dataframe rename operation

df2.rename(columns = {'SibSp':'NumberOfSiblings'}, inplace = True)
print(list(df2.columns))

#Counting survivors by gender:

x1 = pd.crosstab(df2['Survived'],df2['Gender'])
print(x1)

#Count plot for male

sns.countplot(x='male', data=x1)
plt.show()

#Count plot for male

sns.countplot(x='female', data=x1)
plt.show()

#Box plot: Age vs. Pclass And hue should be on Survived.

sns.boxplot(x='Age', y='Pclass', hue='Survived', data=df2)
plt.show()

#Box plot: Age vs. Gender And hue should be on Survived.

sns.boxplot(x='Age', y='Gender', hue='Survived', data=df2)
plt.show()

#THE END
#THANK YOU!