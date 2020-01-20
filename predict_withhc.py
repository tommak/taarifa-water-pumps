# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report

import clean as cl
from clean import HighCardinalityTransformer, inpute_geo
from encoders import MultiClassEncoder, BayesianTargetEncoder

target_encoding = {
    'functional': '0',
    'functional needs repair': '1',
    'non functional': '2'}

target_decoding = {
    '0': 'functional',
    '1': 'functional needs repair',
    '2': 'non functional'}

def read_preprocess(x_path, y_path, target_encoding=None):
    repl = {'construction_year': {0: np.nan},
         'scheme_management': {'None': np.nan},
         'population': {0: np.nan}}

    df = pd.read_csv(x_path)
    if y_path:
        df_y = pd.read_csv(y_path)
        df = pd.merge(df, df_y, left_on='id', right_on='id', how='left')
        if target_encoding:
            df['status_group'] = df['status_group'].replace(target_encoding)
    df = df.replace(repl)

    df.date_recorded = pd.to_datetime(df.date_recorded, format="%Y-%m-%d")
    df['month_recorded'] = df.date_recorded.dt.month
    df['year_recorded'] = df.date_recorded.dt.year
    df['age_recorded'] = df.date_recorded.dt.year - df.construction_year
    df.loc[df.age_recorded < 0, 'age_recorded'] = np.nan
    df.permit = df.permit.replace({True: 'yes', False: 'no'})
    df.public_meeting = df.public_meeting.replace({True: 'yes', False: 'no'})
    
    cl.fe_funder(df, 'funder')
    cl.fe_installer(df, 'installer')
    cl.fe_scheme_name(df, 'scheme_name')

    return df

def train_or_read(clf, X, y, save_path=None, retrain=False):
    if save_path and os.path.exists(save_path) and not retrain:
        print(("Read already trained model"))
        with open(save_path, 'rb') as f:
            clf = pickle.load(f)
    else:
        print("Train model")
        clf.fit(X, y)

        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(clf, f)

    return clf



def make_preprocessor(categorical_features, numeric_features, hc_features):
    numeric_transformer =  SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    hc_transformer = Pipeline(steps=[
        ('high_cardinality', HighCardinalityTransformer(alpha=4)),

        ('imputer2', SimpleImputer(strategy='mean', copy=False)), #, add_indicator=True)),  # proxy for the frequencies over the full dattaset
        ('scaler', StandardScaler()),
        ]) 

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('hc', hc_transformer, hc_features)
        ])

    return preprocessor


def make_preprocessor_bs(categorical_features, numeric_features, hc_features):
    numeric_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    hc_transformer = Pipeline(steps=[
        ('high_cardinality', MultiClassEncoder(BayesianTargetEncoder,
                                            {'columns':hc_features, 'prior_weight': 4}, 'functional')),
        ('imputer2', SimpleImputer(strategy='mean', copy=False)),
        # , add_indicator=True)),  # proxy for the frequencies over the full dattaset
        ('scaler', StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('hc', hc_transformer, hc_features)
        ])

    return preprocessor

# def prepare_data(data, preprocessor=None, categorical_features=None, numeric_features=None ):
#
#     exclude = ['wpt_name',
#                'subvillage',
#                'id',
#                'construction_year',
#                'date_recorded']
#
#     hc_features = ['funder', 'installer', 'lga', 'ward', 'scheme_name',
#                    'region_code', 'district_code']
#
#     ids = data.id
#     X = data.drop(['status_group'] + exclude, axis=1, errors='ignore')
#     y = data.status_group if 'status_group' in data.columns else None
#
#     if not numeric_features:
#         _numeric_features = list(X.select_dtypes(include=np.number).columns)
#         numeric_features = [f for f in _numeric_features if f not in hc_features]
#
#     if not categorical_features:
#         _categorical_features = list(X.select_dtypes(exclude=np.number).columns)
#         categorical_features = [f for f in _categorical_features if f not in hc_features]
#
#     X = X[numeric_features + categorical_features + hc_features].copy()
#
#     if not preprocessor:
#         preprocessor = get_preprocessor(categorical_features, numeric_features,
#                                         hc_features)
#         preprocessor.fit(X, y)
#
#     X = preprocessor.transform(X)
#     return X, y, preprocessor, categorical_features, numeric_features, ids

class DataBuilder():

    def __init__(self, make_preprocessor, exclude, hc_features):
        self.make_preprocessor = make_preprocessor
        self.exclude = exclude
        self.hc_features = hc_features

    def prepare_data(self, data, preprocessor=None, categorical_features=None, numeric_features=None):

        ids = data.id
        X = data.drop(['status_group'] + self.exclude, axis=1, errors='ignore')
        y = data.status_group if 'status_group' in data.columns else None

        if not numeric_features:
            _numeric_features = list(X.select_dtypes(include=np.number).columns)
            numeric_features = [f for f in _numeric_features if f not in self.hc_features]

        if not categorical_features:
            _categorical_features = list(X.select_dtypes(exclude=np.number).columns)
            categorical_features = [f for f in _categorical_features if f not in self.hc_features]

        X = X[numeric_features + categorical_features + self.hc_features].copy()

        if not preprocessor:
            preprocessor = self.make_preprocessor(categorical_features, numeric_features,
                                            self.hc_features)
            preprocessor.fit(X, y)

        X = preprocessor.transform(X)
        return X, y, preprocessor, categorical_features, numeric_features, ids




if __name__ == "__main__":

    path_train_x = '/Users/toma/Documents/Projects/taarifa-water-pumps/data/train_x.csv'
    path_train_y = '/Users/toma/Documents/Projects/taarifa-water-pumps/data/train_y.csv'
    path_test_x = '/Users/toma/Documents/Projects/taarifa-water-pumps/data/test_x.csv'

    exclude = ['wpt_name',
               'subvillage',
               'id',
               'construction_year',
               'date_recorded']

    hc_features = ['funder', 'installer', 'lga', 'ward', 'scheme_name',
                   'region_code', 'district_code']

    data = read_preprocess(path_train_x, path_train_y)
    data_test = read_preprocess(path_test_x, None)

    data['dset'] = 'train'
    data_test['dset'] = 'test'

    data_all = pd.concat([data, data_test], sort=False)
    data_all = inpute_geo(data_all)
    data = data_all.query('dset == "train" ').copy()
    data_test = data_all.query('dset == "test" ').copy()
    data.drop(columns=['dset'], inplace=True)
    data_test.drop(columns=['dset'], inplace=True)

    # builder = DataBuilder(make_preprocessor, exclude, hc_features)
    builder = DataBuilder(make_preprocessor_bs, exclude, hc_features)
    X, y, preprocessor, categorical_features, numeric_features,_ = builder.prepare_data(data)

    # clf_gbm = GradientBoostingClassifier(n_estimators=150, max_depth=4)
    # clf_gbm = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #                                  max_depth=None, max_features=8, max_leaf_nodes=None,
    #                                  min_impurity_decrease=0.0, min_impurity_split=None,
    #                                  min_samples_leaf=2, min_samples_split=2,
    #                                  min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
    #                                  oob_score=False, random_state=None, verbose=0,
    #                                  warm_start=False)

    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                 max_depth=None, max_features=8, max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=2, min_samples_split=2,
                                 min_weight_fraction_leaf=0.0, n_estimators=200,
                                 n_jobs=None, oob_score=False,
                                 verbose=0, warm_start=False)

    clf = train_or_read(clf, X, y, 'clf_rf_bs_0120.pkl' )

    y_train_pred = clf.predict(X)
    print("Train accuracy ", accuracy_score(y, y_train_pred))

    X_test,_ ,_ ,_ ,_ , ids = builder.prepare_data(data_test, preprocessor, categorical_features, numeric_features)
    y_pred = clf.predict(X_test)

    submission = pd.DataFrame(dict(id=ids, status_group=y_pred))
    # submission['status_group'] = submission['status_group'].replace(target_decoding)
    submission.to_csv('submission_bs_0120.csv', index=False)




