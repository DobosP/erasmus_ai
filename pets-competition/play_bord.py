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
import json

random.seed(1)

with open('results.json') as f:
    results = json.load(f)




n_estimators = {}
# Number of features to consider at every split
max_features = {}
# Maximum number of levels in tree
max_depth = {}

# Minimum number of samples required to split a node
min_samples_split = {}

min_samples_leaf = {}

bootstrap = {}


count_result = len(results)





for result in results:

    if result['n_estimators'] not in n_estimators.keys():
        n_estimators[result['n_estimators']] = result['kappa_score']
    else:
        n_estimators[result['n_estimators']] += result['kappa_score']
    if result['max_features'] not in max_features.keys():
        max_features[result['max_features']] = result['kappa_score']
    else:
        max_features[result['max_features']] += result['kappa_score']
    if result['max_depth'] not in max_depth.keys():
        max_depth[result['max_depth']] = result['kappa_score']
    else:
        max_depth[result['max_depth']] += result['kappa_score']
    if result['min_samples_leaf'] not in min_samples_leaf.keys():
        min_samples_leaf[result['min_samples_leaf']] = result['kappa_score']
    else:
        min_samples_leaf[result['min_samples_leaf']] += result['kappa_score']
    if result['min_samples_split'] not in min_samples_split.keys():
        min_samples_split[result['min_samples_split']] = result['kappa_score']
    else:
        min_samples_split[result['min_samples_split']] += result['kappa_score']
    if result['bootstrap'] not in bootstrap.keys():
        bootstrap[result['bootstrap']] = result['kappa_score']
    else:
        bootstrap[result['bootstrap']] += result['kappa_score']


print("n_estimators ",n_estimators)
print("max_features ", max_features)
print("max_depth ", max_depth)
print("min_samples_leaf ", min_samples_leaf)
print("min_samples_split ", min_samples_split)
print("bootstrap", bootstrap)
