# import libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# import data from url

url = 'https://raw.githubusercontent.com/akaakselabhijeet/Data-Science-Datasets-Ver.01/main/Health/FILE_heart.csv'
heart = pd.read_csv(url)

print(heart.tail(6))

# seperate heart_features & heart_target

heart_feature = heart.iloc[:,0:13]
print(heart_feature.head(6))

heart_target = heart.iloc[:,13]
print(heart_target.head(6))

# split dataset in 0.25 test and 0.75 train

feature_train, feature_test, target_train, target_test = train_test_split(heart_feature, heart_target, test_size=0.25, train_size=0.75)

# standard scaler method application

obj = StandardScaler()
train_standardization = obj.fit_transform(feature_train)
print(train_standardization[0:7])

test_standardization = obj.fit_transform(feature_test)
print(test_standardization[0:3])

# use KNN classifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier = classifier.fit(feature_train, target_train)
print(classifier.predict(feature_test))

# derive accuracy scores

predicted_heart_target = classifier.predict(heart_feature)
print(predicted_heart_target[0:10])

accuracy = metrics.accuracy_score(heart_target, predicted_heart_target)
print(accuracy)

classifier = KNeighborsClassifier(n_neighbors=4, metric='minkowski', p=2)
classifier = classifier.fit(feature_train, target_train)
predicted_heart_target = classifier.predict(heart_feature)
accuracy = metrics.accuracy_score(heart_target, predicted_heart_target)
print(accuracy)

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier = classifier.fit(feature_train, target_train)
predicted_heart_target = classifier.predict(heart_feature)
accuracy = metrics.accuracy_score(heart_target, predicted_heart_target)
print(accuracy)

classifier = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2)
classifier = classifier.fit(feature_train, target_train)
predicted_heart_target = classifier.predict(heart_feature)
accuracy = metrics.accuracy_score(heart_target, predicted_heart_target)
print(accuracy)

classifier = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)
classifier = classifier.fit(feature_train, target_train)
predicted_heart_target = classifier.predict(heart_feature)
accuracy = metrics.accuracy_score(heart_target, predicted_heart_target)
print(accuracy)

# THE END
# Abhijeet Das - DS Assignment 10