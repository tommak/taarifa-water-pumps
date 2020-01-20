import functools

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import category_encoders
from sklearn.preprocessing import OneHotEncoder

class BayesianTargetEncoder(BaseEstimator, TransformerMixin):

    """
    Reference: https://www.wikiwand.com/en/Bayes_estimator#/Practical_example_of_Bayes_estimators
    Args:
        columns (list of strs): Columns to encode.
        weighting (int or dict): Value(s) used to weight each prior.
        suffix (str): Suffix used for naming the newly created variables.
    """

    def __init__(self, columns=None, prior_weight=100, suffix='_mean'):
        self.columns = columns
        self.prior_weight = prior_weight
        self.suffix = suffix
        self.prior_ = None
        self.posteriors_ = None

    def fit(self, X, y=None, **fit_params):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        if not isinstance(y, pd.Series):
            raise ValueError('y has to be a pandas.Series')

        X = X.copy()

        # Default to using all the categorical columns
        columns = (
            X.select_dtypes(['object', 'category']).columns
            if self.columns is None else
            self.columns
        )

        names = []
        for cols in columns:
            if isinstance(cols, list):
                name = '_'.join(cols)
                names.append('_'.join(cols))
                X[name] = functools.reduce(
                    lambda a, b: a.astype(str) + '_' + b.astype(str),
                    [X[col] for col in cols]
                )
            else:
                names.append(cols)

        # Compute prior and posterior probabilities for each feature
        X = pd.concat((X[names], y.rename('y')), axis='columns')
        self.prior_ = y.mean()
        self.posteriors_ = {}

        for name in names:
            agg = X.groupby(name)['y'].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']
            pw = self.prior_weight
            self.posteriors_[name] = ((pw * self.prior_ + counts * means) / (pw + counts)).to_dict()

        return self

    def transform(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        for cols in self.columns:

            if isinstance(cols, list):
                name = '_'.join(cols)
                x = functools.reduce(
                    lambda a, b: a.astype(str) + '_' + b.astype(str),
                    [X[col] for col in cols]
                )
            else:
                name = cols
                x = X[name]

            posteriors = self.posteriors_[name]
            X[name] = x.map(posteriors).fillna(self.prior_).astype(float)

        return X


class MultiClassEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, base_encoder, base_encoder_cfg, drop_target_category=None):
        self.base_encoder = base_encoder
        self.base_encoder_cfg = base_encoder_cfg
        self.encoders = []
        self.target_encoder = None
        self.drop_target_category = drop_target_category

    def fit(self, X, y, **kwargs):
        drop_list = [self.drop_target_category] if self.drop_target_category else None
        self.target_encoder = OneHotEncoder(drop=drop_list)
        target = self.target_encoder.fit_transform(y.values.reshape(-1, 1)).toarray()

        nb_targets = target.shape[1]
        for i in range(nb_targets):
            # enc = getattr(category_encoders, self.base_encoder)(**self.base_encoder_cfg)
            enc = self.base_encoder(**self.base_encoder_cfg)

            enc.fit(X, pd.Series(target[:, i]) )
            self.encoders.append(enc)
        return self

    def transform(self, X, y=None):
        target_names = self.target_encoder.get_feature_names()
        l = []
        for i, enc in enumerate(self.encoders):
            X_trf = X.copy()
            X_trf = enc.transform(X_trf, y)
            sfx = '_' + '_'.join(target_names[i].split('_')[1].split())
            X_trf = X_trf.add_suffix(sfx)
            l.append(X_trf)

        return pd.concat(l, axis=1)

