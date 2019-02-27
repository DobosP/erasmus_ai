import random
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import pydot

from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
import score_solution
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

import json

random.seed(1)

df = pd.read_csv('train.csv')

size_data_train = len(df)
truncate_point = int(size_data_train*(float(90)/100))

train_data = df.truncate(before=0, after=truncate_point)
test_data = df.truncate(before=truncate_point, after=size_data_train)



del train_data["Name"]
del train_data["RescuerID"]
del train_data["Description"]
del train_data["PetID"]


adoptionSpeed=train_data['AdoptionSpeed']
del train_data["AdoptionSpeed"]


correct_ans = test_data["AdoptionSpeed"]

del test_data["AdoptionSpeed"]
del test_data["Name"]
del test_data["RescuerID"]
del test_data["Description"]
del test_data["PetID"]
scores_forests = []



# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt']
# Maximum number of levels in tree
max_depths = [int(x) for x in np.linspace(10, 60, num = 6)]
max_depths.append(None)
# Minimum number of samples required to split a node
min_samples_splits = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leafs = [1, 2, 4]
# Method of selecting samples for training each tree
bootstraps = [True, False]


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depths,
               'min_samples_split': min_samples_splits,
               'min_samples_leaf': min_samples_leafs,
               'bootstrap': bootstraps}

# for n_estimator in n_estimators:
#     for max_depth in max_depths:
#         for max_feature in max_features:
#             for min_samples_leaf in min_samples_leafs:
#                 for min_samples_split in min_samples_splits:
#                     for bootstrap in bootstraps:
#                         model = RandomForestClassifier(
#                             n_estimators=n_estimator,
#                             max_depth=max_depth,
#                             max_features=max_feature,
#                             min_samples_leaf=min_samples_leaf,
#                             min_samples_split=min_samples_split,
#                             bootstrap=bootstrap,
#                             random_state=1,
#                             n_jobs=4
#                         )
#                         model.fit(train_data, adoptionSpeed)
#                         ans = model.predict(test_data)
#                         kappa_score = score_solution.kappa(ans, correct_ans)
#
#                         arg_model = {
#                             'kappa_score': kappa_score,
#                             'min_samples_leaf': n_estimator,
#                             'max_depth': max_depth,
#                             'max_features': max_feature,
#                             'min_samples_leaf': min_samples_leaf,
#                             'min_samples_split': min_samples_split,
#                             'bootstrap': bootstrap,
#                         }
#
#                         scores_forests.append(arg_model)
#                         del model
#                         print(arg_model)

total_kappa = 0.0
kappa_eval = make_scorer(cohen_kappa_score)

# for i in range(1,2):
#     rf = RandomForestClassifier()
#     # Random search of parameters, using 3 fold cross validation,
#     # search across 100 different combinations, and use all available cores
#     rf_random = RandomizedSearchCV(
#         estimator = rf, param_distributions = random_grid,
#          n_iter = 100, cv = 3, verbose=2, random_state=1,
#          n_jobs = 4, scoring=kappa_eval
#     )
#     # Fit the random search model
#     rf_random.fit(train_data, adoptionSpeed)
#     best_random = rf_random.best_estimator_
#     print(best_random)
#     ans = best_random.predict(test_data)
#



param_grid = {
    'bootstrap': [True],
    'max_depth': [50, 55, 60, 45, 40],
    'max_features': ['log2'],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [5, 4, 6],
    'n_estimators': [700, 650, 630, 750]
}

rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=40, max_features='log2', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=6,
            min_weight_fraction_leaf=0.0, n_estimators=630, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
#
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
#                           cv = 3, n_jobs = 4, verbose = 2,
#                           scoring=kappa_eval)
# Fit the random search model
rf.fit(train_data, adoptionSpeed)

ans = rf.predict(test_data)


#
#
# for e in float_ans:
#     ans.append(int(round(e)))



kappa_score = score_solution.kappa(ans, correct_ans)
print(kappa_score)
#
#
# kappa_eval = make_scorer(cohen_kappa_score)
