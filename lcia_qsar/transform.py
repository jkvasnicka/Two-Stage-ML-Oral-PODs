'''This module contains custom transformers following the scikit-learn API.

Notes
-----
Currently, these custom transformers only work with pandas objects as the
inputs. The pandas.DataFrame, in particular, facilitates the use of 
sklearn.compose.ColumnTransformer, e.g., to pass through discrete-valued 
features for some transformations (e.g., center/scale) based on the columns.

Reference
---------
https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects
'''

import numpy as np
from sklearn.base import (
    OneToOneFeatureMixin,
    TransformerMixin,
    BaseEstimator
)
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

import features

#region: MissingValuesSelector
class MissingValuesSelector(TransformerMixin, BaseEstimator):
    '''Discard features with proportion of missing values (NaN) > threshold.

    See Also
    --------
    sklearn.feature_selection.SelectorMixin
        This pandas API not fully implemented yet.
    sklearn.feature_selection.VarianceThreshold
        For example of the API with SelectorMixin.
    '''
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def fit(self, X, y=None):
        '''
        '''
        self.proportions_missing_ = X.isna().mean() 
        
        self.n_features_in_ = X.shape[1]
        
        return self 

    def _get_support_mask(self):
        '''
        '''
        check_is_fitted(self)

        return self.proportions_missing_ <= self.threshold
    
    def transform(self, X):
        '''
        '''
        mask = self._get_support_mask()
        return X.loc[:, mask]
#endregion

# FIXME: Use one-to-one mixin instead of FeatureNameSupport?
# Change to median scaler only.
#region: MedianScaler
class MedianScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    '''Standardize features and scale to unit variance.

    Uses the median and median absolute deviation to center and scale.

    See Also
    --------
    sklearn.preprocessing.StandardScaler
        Uses mean/SD instead of median/MAD.
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        '''Set the center_ and scale_ attributes.
        '''
        self.center_ = X.quantile(0.5)
        self.scale_ = features.median_absolute_deviation(X)
        
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns, dtype=object)
                                      
        return self

    def transform(self, X):
        '''Center and scale the continuous features and return a copy.
        '''
        check_is_fitted(self)
        assert self.n_features_in_ == X.shape[1]
        assert (self.center_.index == self.scale_.index).all()
        
        return features.center_scale(X, self.center_, self.scale_)
#endregion

# TODO: May no longer be used.
#region: Log10Transformer
class Log10Transformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    '''Selectively apply a log10-transformation to features.

    By default, features will be transformed if they are not significantly 
    left-skewed.

    Parameters
    ----------
    alpha : int (optional)
        Statistical significance level for scipy.stats.skewtest. Features with 
        p-values less than this value are deemed "significant."
    '''
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def fit(self, X, y=None):
        '''Identify which features to transform.

        Set the features_to_transform_ attribute, which is a list of str
        corresponding to the columns/features.
        '''
        left_skewed = features.skewed_columns(
            X, alpha=self.alpha, alternative='less') 
        self.features_to_transform_ = [
            c for c in X if c not in left_skewed]
        
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns, dtype=object)
                                      
        return self
    
    def transform(self, X):
        '''Apply a log10-transformation to those features specfied in fit().
        '''
        check_is_fitted(self)
        assert self.n_features_in_ == X.shape[1]

        X.loc[:, self.features_to_transform_] = (
            np.log10(X[self.features_to_transform_]))

        return X
#endregion

#region: select_columns_without_pattern
class select_columns_without_pattern:
    '''Create a callable to select columns to be used with 
    sklearn.compose.ColumnTransformer.

    This callable returns columns that do NOT contain the specified regex 
    pattern.
    '''
    def __init__(self, pattern):
        self.pattern = pattern
        
    def __call__(self, X):
        '''Callable for column selection to be used by a ColumnTransformer.
        '''
        cols = X.columns
        if self.pattern is not None:
            cols = cols[~cols.str.contains(self.pattern, regex=True)]
        return list(cols)
#endregion