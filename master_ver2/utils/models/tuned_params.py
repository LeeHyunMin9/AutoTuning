import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, RANSACRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Define Distribution =================================================
class UniformDistribution():
    def __init__(self, low : float, high : float, log : bool = False):
        self.low = low
        self.high = high
        self.log = log

    def get_base(self):
        """ Get distributions from base libraries such as scipy, numpy, etc. """
        if self.log:
            return np.random.uniform(np.log(self.low), np.log(self.high))
        
        return np.random.uniform(self.low, self.high)

    def __repr__(self):
        return f"UniformDistribution(low={self.low}, high={self.high}, log={self.log})"
    
class IntUniformDistribution():
    def __init__(self, low : int, high : int):
        self.low = low
        self.high = high

    def get_base(self):
        return np.random.randint(self.low, self.high)

    def __repr__(self):
        return f"IntUniformDistribution(low={self.low}, high={self.high})"


# Define Configuration of Estimators

AVAILABLE_ESTIMATORS = {
            'en' : {'model' : ElasticNet, },
            'svm' : {'model' : SVR},
            'ransac' : {'model' : RANSACRegressor},
            'knn' : {'model' : KNeighborsRegressor},
            'rf' : {'model' : RandomForestRegressor},
            'xgboost' : {'model' : XGBRegressor},
            'lightgbm' : {'model' : LGBMRegressor}
        }

for key in AVAILABLE_ESTIMATORS.keys():
    AVAILABLE_ESTIMATORS[key]['args'] = dict()
    AVAILABLE_ESTIMATORS[key]['tune_grid'] = dict()
    AVAILABLE_ESTIMATORS[key]['tune_distributions'] = dict()
    AVAILABLE_ESTIMATORS[key]['tune_args'] = dict()


AVAILABLE_ESTIMATORS['en']['tune_grid'] = {
    "alpha" : np.arange(0.01, 10, 0.01),
    "l1_ratio" : np.arange(0.01, 1, 0.001),
    "fit_intercept" : [True, False]
} 

AVAILABLE_ESTIMATORS['en']['tune_distributions'] = {
    "alpha" : UniformDistribution(0,1),
    "l1_ratio" : UniformDistribution(0.01 , 0.999999999)
}

AVAILABLE_ESTIMATORS['svm']['tune_grid'] = {
    'C' : np.arange(0, 10, 0.001),
    'epsilon' : [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9, 2.0]
}

AVAILABLE_ESTIMATORS['svm']['tune_distributions'] = {
    'C' : UniformDistribution(0, 10),
    'epsilon' : UniformDistribution(1.0, 2.0)
}

AVAILABLE_ESTIMATORS['ransac']['args'] = {
    "random_state" : 42
}

AVAILABLE_ESTIMATORS['ransac']['tune_grid'] = {
    "min_samples" : np.arange(0.1, 1, 0.05),
    "max_trials" : np.arange(1, 20, 1),
    "max_skips" : np.arange(1, 20, 1),
    "stop_n_inliers" : np.arange(1, 25, 1),
    "stop_probability" : np.arange(0.1, 1, 0.05)
}

AVAILABLE_ESTIMATORS['ransac']['tune_distributions'] = {
    "min_samples" : UniformDistribution(0.1, 1),
    "max_trials" : IntUniformDistribution(1, 20),
    "max_skips" : IntUniformDistribution(1, 20),
    "stop_n_inliers" : IntUniformDistribution(1, 25),
    "stop_probability" : UniformDistribution(0.1, 1)
}

AVAILABLE_ESTIMATORS['knn']['tune_grid'] = {
    "n_neighbors" : np.arange(1, 51, 1),
    "weights" : ['uniform'],
    "metric" : ['euclidean', 'manhattan', 'minkowski'],
}

AVAILABLE_ESTIMATORS['knn']['tune_distributions'] = {
    "n_neighbors" : IntUniformDistribution(1, 51),
}


AVAILABLE_ESTIMATORS['rf']['tune_grid'] = {
    "n_estimators" : np.arange(10, 300, 10),
    "max_depth" : np.arange(1, 11, 1),
    "min_impurity_decrease" : np.array([0, 0.0001, 0.0005, 0.001, 0.005 ,0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
    "max_features" : ['auto', 'sqrt', 'log2'],
    "bootstrap" : [True, False],

    "min_samples_split" : np.array([2, 5, 7, 9, 10]),
    "min_samples_leaf" : np.array([2, 3, 4, 5, 6]),
}

AVAILABLE_ESTIMATORS['rf']['tune_distributions'] = {
    "n_estimators" : IntUniformDistribution(10, 300),
    "max_depth" : IntUniformDistribution(1, 11),
    "min_impurity_decrease" : UniformDistribution(1e-6, 0.5, log = True),
    "max_features" : UniformDistribution(0.4, 1),

    "min_samples_split" : IntUniformDistribution(2, 10),
    "min_samples_leaf" : IntUniformDistribution(2, 6),
}

AVAILABLE_ESTIMATORS['xgboost']['args'] = {
    "device" : "cpu",
    "random_state" : 42,
    "booster" : "gbtree",   
    "tree_method" : "auto",
}

AVAILABLE_ESTIMATORS['xgboost']['tune_grid'] = {
    "learning_rate" : np.array([1e-6, 1e-5, 1e-4, 1e-3, 5e-3 ,0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]),
    "n_estimators" : np.arange(10, 300, 10),
    "subsample" : np.array([0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    "max_depth" : np.arange(1, 11, 1),
    "colsample_bytree" : np.array([0.5, 0.7, 0.9, 1.0]),
    "min_child_weight" : np.array([1, 2, 3, 4]),
    "reg_alpha" : np.array([1e-6, 1e-5, 1e-4, 1e-3, 5e-3 ,0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]),
    "reg_lambda" : np.array([1e-6, 1e-5, 1e-4, 1e-3, 5e-3 ,0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]),
    "scale_pos_weight" : np.arange(0, 50, 0.1),
}

AVAILABLE_ESTIMATORS['xgboost']['tune_distributions'] = {
    "learning_rate" : UniformDistribution(1e-6, 0.5, log = True),
    "n_estimators" : IntUniformDistribution(10, 300),
    "subsample" : UniformDistribution(0.2, 1.0),
    "max_depth" : IntUniformDistribution(1, 11),
    "colsample_bytree" : UniformDistribution(0.5, 1.0),
    "min_child_weight" : IntUniformDistribution(1, 4),
    "reg_alpha" : UniformDistribution(1e-6, 10.0, log = True),
    "reg_lambda" : UniformDistribution(1e-6, 10.0, log = True),
    "scale_pos_weight" : UniformDistribution(0, 50),
}


AVAILABLE_ESTIMATORS["lightgbm"]["tune_grid"] = {
    "num_leaves" : np.array([2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 256]),
    "learning_rate" : np.array([1e-6, 1e-5, 1e-4, 1e-3, 5e-3 ,0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]),
    "n_estimators" : np.arange(10, 300, 10),
    "min_split_gain" : np.arange(0, 1.0, 0.1),
    "reg_alpha" : np.array([1e-6, 1e-5, 1e-4, 1e-3, 5e-3 ,0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]),
    "reg_lambda" : np.array([1e-6, 1e-5, 1e-4, 1e-3, 5e-3 ,0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]),
    "feature_fraction" : np.arange(0.4, 1.0, 0.1),
    "bagging_fraction" : np.arange(0.4, 1.0, 0.1),
    "bagging_freq" : np.arange(0, 8, 1),
    "min_child_samples" : np.arange(1, 100, 5),
}

AVAILABLE_ESTIMATORS["lightgbm"]["tune_distributions"] = {
    "num_leaves" : IntUniformDistribution(2, 256),
    "learning_rate" : UniformDistribution(1e-6, 0.5, log = True),
    "n_estimators" : IntUniformDistribution(10, 300),
    "min_split_gain" : UniformDistribution(0, 1.0),
    "reg_alpha" : UniformDistribution(1e-6, 10.0, log = True),
    "reg_lambda" : UniformDistribution(1e-6, 10.0, log = True),
    "feature_fraction" : UniformDistribution(0.4, 1.0),
    "bagging_fraction" : UniformDistribution(0.4, 1.0),
    "bagging_freq" : IntUniformDistribution(0, 7),
    "min_child_samples" : IntUniformDistribution(1, 100),
}