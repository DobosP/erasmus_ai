import random
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import pydot

from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from sklearn import preprocessing
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, KFold

import scipy as sp
from collections import Counter
from functools import partial
from math import sqrt
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from joblib import Parallel, delayed
from tqdm import tqdm, tqdm_notebook


from typing import TypeVar, List, Dict, Tuple
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')



# train_data = pd.read_csv('../input/data-v64/all_train_data_V6_pure_breed_V4.csv')
# test_data = pd.read_csv('../input/data-v64/all_test_data_V6_pure_breed_V4.csv')

train_data = pd.read_csv('../input/data-v8-complete/all_train_data_V8_final.csv')
test_data = pd.read_csv('../input/data-v8-complete/all_test_data_V8_final.csv')



test_ids = test_data[['PetID']]


del train_data["Name"]
del train_data["PetID"]


del test_data["Name"]
del test_data["PetID"]


def impact_coding(
    data: PandasDataFrame,
    feature: str, target='y'
) -> Tuple[pd.Series, PandasDataFrame, PandasDataFrame]:
    '''Crate a new feature that have a dependecie with  a target.
    This feature increase the importance of an existing feature.
    '''
    n_folds = 20
    n_inner_folds = 10
    impact_coded = pd.Series()

    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)
    kf = KFold(n_splits=n_folds, shuffle=True)
    oof_mean_cv = pd.DataFrame()
    split = 0
    for infold, oof in kf.split(data[feature]):
            impact_coded_cv = pd.Series()
            kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
            inner_split = 0
            inner_oof_mean_cv = pd.DataFrame()
            oof_default_inner_mean = data.iloc[infold][target].mean()
            for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):
                # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
                oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()
                impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(
                            lambda x: oof_mean[x[feature]]
                                      if x[feature] in oof_mean.index
                                      else oof_default_inner_mean
                            , axis=1))

                # Also populate mapping (this has all group -> mean for all inner CV folds)
                inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
                inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
                inner_split += 1

            # Also populate mapping
            oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
            oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
            split += 1

            impact_coded = impact_coded.append(data.iloc[oof].apply(
                            lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
                                      if x[feature] in inner_oof_mean_cv.index
                                      else oof_default_mean
                            , axis=1))

    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean



# This features will have a grater inpact in predictions
categorical_features = [
    'Color1', 'main_breed_Type',
    'second_breed_Type', 'Color2',
    'Type', 'Health', 'sen1_score_Mean'
]
# categorical_features_lgb =['Age',
#     'State', 'main_breed_BreedName', 'Fee', 'Color1', 'Breed2', 'Color2',
#        'MaturitySize', 'Vaccinated',
#        'Color3', 'PureBreed', 'Health', 'second_breed_Type',
#        'main_breed_Type', 'Type']

impact_coding_map = {}
for f in categorical_features:
    print("Impact coding for {}".format(f))
    train_data["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(
        train_data, f, target="AdoptionSpeed"
    )
    impact_coding_map[f] = (impact_coding_mapping, default_coding)
    mapping, default_mean = impact_coding_map[f]
    test_data["impact_encoded_{}".format(f)] = test_data.apply(
        lambda x: mapping[x[f]] if x[f] in mapping
                                else default_mean, axis=1
    )



# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

    class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return -cohen_kappa_score(y, preds, weights='quadratic')

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return preds

    def coefficients(self):
        return self.coef_['x']


lgb_params = {
    'application': 'regression',
    'boosting': 'gbdt',
    'metric': 'rmse',
    'num_leaves': 70,
    'max_depth': 9,
    'learning_rate': 0.01,
    'bagging_fraction': 0.85,
    'feature_fraction': 0.8,
    'min_split_gain': 0.02,
    'min_child_samples': 150,
    'min_child_weight': 0.02,
    'lambda_l2': 0.0475,
    'verbosity': -1,
    'data_random_seed': 17,
}




cat_params = {
          'depth': 9,
          'eta': 0.03,
          'task_type' :"GPU",
          'random_strength': 1.5,
          'loss_function': 'RMSE',
          'reg_lambda': 6,
          'od_type': 'Iter',
          'border_count': 128,
          'bootstrap_type' : "Bayesian",
          'random_seed': 123455,
          'use_best_model':True,
          'eval_metric': 'RMSE',
}

xgb_params = {
    'eval_metric': 'rmse',
    'seed': 1337,
    'eta': 0.0123,
    'subsample': 0.8,
    'colsample_bytree': 0.85,
    'tree_method': 'gpu_hist',
    'device': 'gpu',
    'silent': 1,
}




def run_xgb(
    params: Dict, X_train: PandasDataFrame,
    X_test: PandasDataFrame
) -> Tuple[xgb.Booster,PandasDataFrame,PandasDataFrame]:
    """CV train xgb Booster.

    Args:
        params: Params for Booster.
        X_train: Training dataset.
        X_test: Testing dataset.

    Returns:
        model: Trained model.
        oof_train_xgb:  Training CV predictions.
        oof_test_xgb:  Testing CV predictions.
    """

    n_splits = 10
    verbose_eval = 25
    num_rounds = 60000
    early_stop = 500

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)

    oof_train_xgb = np.zeros((X_train.shape[0]))
    oof_test_xgb = np.zeros((X_test.shape[0], n_splits))

    i = 0

    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):

        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

        oof_train_xgb[valid_idx] = valid_pred
        oof_test_xgb[:, i] = test_pred

        i += 1
    return model, oof_train_xgb, oof_test_xgb


def run_cat(
    params: Dict,
    X_train: PandasDataFrame,
    X_test: PandasDataFrame
)  -> Tuple[cat.CatBoost, PandasDataFrame, PandasDataFrame]:
    """CV train cat Booster.

    Args:
        params: Params for Booster.
        X_train: Training dataset.
        X_test: Testing dataset.

    Returns:
        model: Trained model.
        oof_train_cat:  Training CV predictions.
        oof_test_cat:  Testing CV predictions.
    """

    early_stop = 1000
    verbose_eval = 25
    num_rounds = 10000
    n_splits = 10

    kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)

    oof_train_cat = np.zeros((X_train.shape[0]))
    oof_test_cat = np.zeros((X_test.shape[0], n_splits))


    i = 0
    for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].values):

        X_tr = X_train.iloc[train_index, :]
        X_val = X_train.iloc[valid_index, :]

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        train_pool = cat.Pool(X_tr, y_tr)
        test_pool = cat.Pool(X_val, y_val)

        #train the model
        model = cat.train(
            pool=train_pool,
            params=params,
            iterations=num_rounds,
            eval_set=test_pool,
            early_stopping_rounds=early_stop,
            verbose=verbose_eval)


        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        oof_train_cat[valid_index] = val_pred
        oof_test_cat[:, i] = test_pred

        i += 1

    return model, oof_train_cat, oof_test_cat



def run_lgb(
    params: Dict, X_train: PandasDataFrame,
    X_test: PandasDataFrame
) -> Tuple[lgb.Booster,PandasDataFrame,PandasDataFrame]:
    """CV train lgb Booster.

    Args:
        params: Params for Booster.
        X_train: Training dataset.
        X_test: Testing dataset.

    Returns:
        model: Trained model.
        oof_train_lgb:  Training CV predictions.
        oof_test_lgb:  Testing CV predictions.
    """

    early_stop = 1000
    verbose_eval = 25
    num_rounds = 10000
    n_splits = 10



    kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)


    oof_train_lgb = np.zeros((X_train.shape[0]))
    oof_test_lgb = np.zeros((X_test.shape[0], n_splits))

    i = 0
    for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].values):

        X_tr = X_train.iloc[train_index, :]
        X_val = X_train.iloc[valid_index, :]

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)



        d_train = lgb.Dataset(X_tr, label=y_tr)
        d_valid = lgb.Dataset(X_val, label=y_val)
        watchlist = [d_train, d_valid]

        model = lgb.train(
            params,
            train_set=d_train,
            num_boost_round=num_rounds,
            valid_sets=watchlist,
            verbose_eval=verbose_eval,
            early_stopping_rounds=early_stop,
#             categorical_feature=categorical_features_lgb
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)

        oof_train_lgb[valid_index] = val_pred
        oof_test_lgb[:, i] = test_pred

        i += 1



    return model, oof_train_lgb, oof_test_lgb



cat_model, oof_train_cat, oof_test_cat = run_cat(cat_params, train_data, test_data)
lgb_model, oof_train_lgb, oof_test_lgb = run_lgb(lgb_params, train_data, test_data)
xgb_model, oof_train_xgb, oof_test_xgb = run_xgb(xgb_params, train_data, test_data)



importance_type= "split" # "gain"
idx_sort_lgb = np.argsort(lgb_model.feature_importance(importance_type=importance_type))[::-1]
names_sorted_lgb = np.array(lgb_model.feature_name())[idx_sort_lgb]


idx_sort_cat = np.argsort(cat_model.feature_importances_)[::-1]
names_sorted_cat = np.array(cat_model.feature_names_)[idx_sort_cat]

dict_importance_xgb = xgb_model.get_score(importance_type='total_gain')
names_sorted_xgb = []
for key, value in sorted(dict_importance_xgb.items(), key = lambda x:x[1])[::-1]:
    names_sorted_xgb.append(key)


imp_pred = {
    "lgb": list(names_sorted_lgb),
    "xgb": list(names_sorted_xgb),
    "cat": list(names_sorted_cat),
}

def get_importance(imp_pred: Dict) -> List:
    """Agregate the  importance predictions.

    Args:
        imp_pred: 3 predictions.

    Returns:
        solution: Sorted features by importance in the combine model
    """
    set_features =  imp_pred['lgb'] and imp_pred['cat'] and imp_pred['xgb']

    MAP = {}

    for element in set_features:

        index1 = 1 + imp_pred['lgb'].index(element)
        index2 = 1 + imp_pred['cat'].index(element)
        index3 = 1 + imp_pred['xgb'].index(element)
        index = index1 + index2 + index3
        MAP[element] = index
    solution = []
    for key,val in sorted(MAP.items(), key =lambda x:x[1]):

        solution.append(key)

    return solution


feature_importance =  get_importance(imp_pred)

print(feature_importance)


def predict_optimiser(
    oof_train: PandaDataFrame,
    train_ans: List,
    oof_test: PandaDataFrame,
    coef: List
) -> List:
    """Optimse a prediction.

    By using the results form CV training
    compute boundaries for prediction classes.

    Args:
        oof_triain: Predictions for training data.
        train_ans: Correct values for prediction taget.
        oof_test: Predictions for test data.
        coef: List of parameters for optimisation.
    Returns:
        test_predictions: Optimised prediction.
    """

    optR = OptimizedRounder()
    optR.fit(oof_train, train_ans)
    coefficients = optR.coefficients()
    if len(coef) != 0:
        coefficients[0] = coef[0]
        coefficients[1] = coef[1]
        coefficients[3] = coef[3]
    valid_pred = optR.predict(oof_train, coefficients)
    qwk = quadratic_weighted_kappa(train_ans, valid_pred)
    print(f"qwk score: {qwk}")
    coefficients_ = coefficients.copy()
    print(coefficients_)
    if len(coef) != 0:
        coefficients_[0] = coef[0]
        coefficients_[1] = coef[1]
        coefficients_[3] = coef[3]
    train_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)
    print(f'train pred distribution: {Counter(train_predictions)}')
    test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)
    print(f'test pred distribution: {Counter(test_predictions)}')
    return test_predictions


# This coefficients provide good in practice
# lgb_coef = [1.645, 2.115, 0, 2.84]
lgb_coef = [1.66, 2.13, 0, 2.85]
lgb_ans = predict_optimiser(oof_train_lgb, train_data['AdoptionSpeed'].values, oof_test_lgb, lgb_coef)
xgb_coef = [1.66, 2.13, 0, 2.85]
xgb_ans = predict_optimiser(oof_train_xgb, train_data['AdoptionSpeed'].values, oof_test_xgb, xgb_coef)
cat_coef = [1.66, 2.13, 0, 2.85]
cat_ans = predict_optimiser(oof_train_cat, train_data['AdoptionSpeed'].values, oof_test_cat, cat_coef)


predictions = {
    "lgb": lgb_ans,
    "xgb": xgb_ans,
    "cat": cat_ans

}
weights  = {
    "lgb": 1,
    "xgb": 1,
    "cat": 1
}


def weighted_voting(predictions: Dict, weights: Dict) -> List:
    """Weighted voting system.

    Combine models in a weghted manner.

   Args:
       predictions: Models predictions.
       weights: Models weights.
    Retruns:
       final_ans: Agregated prediction.
    """
    final_ans = []
    index = 0

    for index in range(len(predictions['xgb'])):
        prediction = {0:0, 1:0, 2: 0, 3: 0, 4:0 }
        for model in predictions.keys():
            prediction[predictions[model][index]] += weights[model]

        max_vote = 0
        sols_ans = []
        sol_ans = 0
        for ans, vote in prediction.items():
            if vote == max_vote and vote != 0:
                sols_ans.append(ans)
            if vote > max_vote:
                sols_ans = [ans]
                max_vote = vote
                sol_ans = ans

        if len(sols_ans) > 2:
            final_ans.append(sol_ans)
        else:
            sols_ans.sort()
            final_ans.append(sols_ans[int(len(sols_ans)/2)])
    return final_ans



final_ans =  weighted_voting(predictions, weights)

submission = pd.DataFrame({'PetID': test_ids['PetID'].values,
                           'AdoptionSpeed': final_ans
                          })
# submission.head()
submission.to_csv('submission.csv', index=False)
