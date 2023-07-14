'''Provide support for instantiating machine-learning estimators.
'''

from sklearn.linear_model import (
    LinearRegression,
    Ridge, 
    Lasso, 
    ElasticNet,
    Lars,
    LassoLars,
    OrthogonalMatchingPursuit,
    BayesianRidge,
    ARDRegression,
    RANSACRegressor,
    TheilSenRegressor,
    HuberRegressor,
    SGDRegressor,
    QuantileRegressor
)
from sklearn.kernel_ridge import KernelRidge   
from sklearn.svm import (
    LinearSVR,
    NuSVR,
    SVR
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

## Define a container to choose among all estimators.
_estimator_classes = [
    LinearRegression,
    Ridge, 
    Lasso, 
    ElasticNet,
    Lars,
    LassoLars,
    OrthogonalMatchingPursuit,
    BayesianRidge,
    ARDRegression,
    RANSACRegressor,
    TheilSenRegressor,
    HuberRegressor,
    SGDRegressor, 
    QuantileRegressor,
    KernelRidge,   
    LinearSVR,
    NuSVR,
    SVR,
    GaussianProcessRegressor,
    PLSRegression,
    DecisionTreeRegressor,
    RandomForestRegressor, 
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    XGBRegressor,
    MLPRegressor
]
class_for_name = {ec.__name__ : ec for ec in _estimator_classes}