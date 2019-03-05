############################################################################################################
# About: Feature importance extraction based on a random forest
# About the variables:
# df_train - all train data (what you load locally)
# submission_test - all test data
# adoptionSpeed - holds adoption speed values from df_train
# train - training data set for model evaluation (80% of df_train)
# test - testing data set for model evaluation (20% of df_train)
# adoptionSpeed_train, adoptionSpeed_test - corresponding values of the adooption speed, used for model evaluation
# feature_train - only n most important features (columns) from train 
# feature_test - only n most important features (columns) from test
# max_features - set how many features to use to create feature_train and feature_test

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import pydot

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

# Set the correct path for yourself:

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Correct the relative path for yourself
df_train = pd.read_csv('../input/all-data/all_train_data.csv')
submission_test = pd.read_csv('../input/all-data/all_test_data.csv')

# sub is used to write the results down
sub = submission_test['PetID']

# Remove object data types from the data
################################################
# If you work with V2 or V3 of the data, there is not column 'Unnamed: 0', so manually remove it from the two lines below
################################################
df_train.drop(['Unnamed: 0','Name', 'RescuerID', 'Description', 'PetID'], axis=1, inplace=True)
submission_test.drop(['Unnamed: 0','Name', 'RescuerID', 'Description', 'PetID'], axis=1, inplace=True)
df_train.info()
df_train.describe()

# extract adoptionSpeed in a separate variable, remove it from the data set for model training/evaluation
df_train = df_train.dropna()
adoptionSpeed=df_train['AdoptionSpeed']
del df_train["AdoptionSpeed"]
df_train.info()

# Normalize the data
columns = df_train.columns
df_train = pd.DataFrame(preprocessing.scale(df_train))
df_train.columns = columns
df_train.describe()

# Separate the training data set into train (80% df_train) and test (20% df_train)
train, test, adoptionSpeed_train, adoptionSpeed_test = train_test_split(df_train, adoptionSpeed, test_size=0.2)
model = RandomForestClassifier(n_estimators=500, criterion='entropy') # n_estimators number of decision trees

# Train a random forest classifier for feature importance for all pets
model.fit(train, adoptionSpeed_train)

importances = model.feature_importances_
std = np.std([t.feature_importances_ for t in model.estimators_],
             axis=0)
index = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, index[f], importances[index[f]]))
    
labels = list(train.columns.values)
ordered_labels = [labels[i] for i in index]

# Plot the feature importances of the forest
fig, ax = plt.subplots()
fig.canvas.draw()
plt.title("Feature importances for all pets")
plt.bar(range(train.shape[1]), importances[index],
       color="r", yerr=std[index], align="center")
plt.xticks(range(train.shape[1]), index, rotation='vertical')
plt.xlim([-1, train.shape[1]])
ax.set_xticklabels(ordered_labels)
plt.show()

# For the submission test data, I first set all nans to 0 and then normalize it
from numpy import *
all_nans = isnan(submission_test)
submission_test[all_nans] = 0
submission_columns = submission_test.columns
submission_test = pd.DataFrame(preprocessing.scale(submission_test))
submission_test.columns = submission_columns

# Extract data sets with max_features most important features
feature_train = train.iloc[:, index[0]]
feature_test = test.iloc[:, index[0]]
feature_submission = submission_test.iloc[:, index[0]]

max_features = 15

for i in range(1, max_features):
    subset1 = train.iloc[ : ,index[i]]
    subset2 = test.iloc[ : ,index[i]]
    subset3 = submission_test.iloc[:, index[i]]
    feature_train = pd.concat([feature_train, subset1], axis=1)
    feature_test = pd.concat([feature_test, subset2], axis=1)
    feature_submission = pd.concat([feature_submission, subset3], axis=1)
