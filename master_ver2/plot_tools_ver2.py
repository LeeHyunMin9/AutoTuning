# Compare the scikit-learn pipeline, data preprocessing, hyper parameter optimization with pycaret pipeline, data preprocessing, hyper parameter optimization, repectively.

import sys
import numpy as np
import pandas as pd
import json
import os
import logging
import time
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Union

#from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline
from scipy import sparse
from pycaret.regression import *


from utils.data import df_shrink_dtypes
from utils.transformer import TransformerWrapperWithInverse, TargetTransformer, TransformerWrapper
from utils.models.tuned_params import AVAILABLE_ESTIMATORS

# Group of variable types for isinstance
SEQUENCE = (list, tuple, np.ndarray, pd.Series)

# Variable types for type hinting
SEQUENCE_LIKE = Union[SEQUENCE]
DATAFRAME_LIKE = Union[dict, list, tuple, np.ndarray, sparse.spmatrix, pd.DataFrame]
TARGET_LIKE = Union[int, str, list, tuple, np.ndarray, pd.Series]


if len(sys.argv) == 1:
    
    EXP_CONFIG = {
        'motor_type' : "Hyeon_Seo",
        'current_inertia' : 9,
        'acceleration_time' : 0.010,
        'exp_num' : 0
    }

else:
    EXP_CONFIG = {
        'motor_type' : sys.argv[1],
        'current_inertia' : int(sys.argv[2]),
        'acceleration_time' : float(sys.argv[3]),
        'exp_num' : int(sys.argv[4])
    }

DATA_DIR = f'D:\\001.Developement of Company Work\\features\\{EXP_CONFIG["motor_type"]}\\A.T_{EXP_CONFIG["acceleration_time"]:.3f}_R.I_{EXP_CONFIG["current_inertia"]:04d}\\trial_{EXP_CONFIG["exp_num"]}'

LOGGER = logging.getLogger()



# Data Preprocessing
data = pd.read_excel(f'{DATA_DIR}\\dataframe.xlsx', index_col = 0)

# # train - test split    
# train, test = train_test_split(data, test_size = 0.2, random_state = 42)

# PyCaret 환경을 설정합니다
# 1. data 를 구성함
# 2. Preprocessing Class를 구성함
class Preprocessor:
    """Class for all standard transformation steps"""
    def __init__(self, seed = 42):
        self.logger = LOGGER 
        self.seed = seed

    def _prepare_data(self, data, target):

        self.logger.info(f"Data Shape: {data.shape}")

        # Duplicate Check
        if len(set(data.index)) != len(data):
            self.logger.warning("Data has duplicated index")
            raise ValueError("Data has duplicated index")
        
        # Prepare Target Columns
        if isinstance(target, (list, tuple, np.ndarray, pd.Series)):
            # Multi-Target(Iterable)
            if not isinstance(target, pd.Series):
                
                if np.array(target).ndim != 1:
                    self.logger.warning("Target is not 1-D array")
                    raise ValueError("Target is not 1-D array")

                if len(self.data) != len(target):
                    self.logger.warning(f"Data and Target have different length, got len(data) = {len(data)} and len(target) = {len(target)}")
                    raise ValueError(f"Data and Target have different length, got len(data) = {len(data)} and len(target) = {len(target)}")
                
                target = pd.Series(target, index = data.index)

            elif not self.data.index.equals(target.index):
                self.logger.warning("Data and Target have different index")
                raise ValueError("Data and Target have different index")

        elif isinstance(target, str):
            if target not in data.columns:
                self.logger.warning(f"Target {target} is not in the data columns")
                raise ValueError(f"Target {target} is not in the data columns")
            
            X, y = data.drop(columns = target, axis =1), data[target]

        elif isinstance(target, int):
            X, y = data.drop(columns = data.columns[target], axis =1), data[data.columns[target]]

        else: # None Target
            return df_shrink_dtypes(X)

        return df_shrink_dtypes( pd.concat([X, y], axis = 1))

    def _set_index(self, df):
        """Assign an index to the dataframe."""
        self.logger.info("Set up index.")

        target = df.columns[-1]

        if getattr(self, "index", True) is True:  # True gets caught by isinstance(int)
            pass

        elif self.index is False:
            df = df.reset_index(drop=True)

        elif isinstance(self.index, int):
            if -df.shape[1] <= self.index <= df.shape[1]:
                df = df.set_index(df.columns[self.index], drop=True)
            else:
                raise ValueError(
                    f"Invalid value for the index parameter. Value {self.index} "
                    f"is out of range for a dataset with {df.shape[1]} columns."
                )
        elif isinstance(self.index, str):
            if self.index in df:
                df = df.set_index(self.index, drop=True)
            else:
                raise ValueError(
                    "Invalid value for the index parameter. "
                    f"Column {self.index} not found in the dataset."
                )

        if df.index.name == target:
            raise ValueError(
                "Invalid value for the index parameter. The index column "
                f"can not be the same as the target column, got {target}."
            )

        if df.index.duplicated().any():
            raise ValueError(
                "Invalid value for the index parameter. There are duplicate indices "
                "in the dataset. Use index=False to reset the index to RangeIndex."
            )

        return df

    def _prepare_train_test(self, test_size, test_data):
        """Split the data into training and testing sets."""
        self.logger.info("Split the data into training and testing sets.")

        if test_data is None:
            self.train, self.test = train_test_split(self.data, test_size = test_size, random_state = self.seed)
            self.data = self._set_index(pd.concat([self.train, self.test], axis = 0))
            self.idx = [self.data.index[: len(self.train)], self.data.index[-len(self.test) : ]]

        else: # test_data is already given
            test_data = self._prepare_data(test_data, target = self.target_param)
            self.data = self._set_index(pd.concat([self.data, test_data], axis = 0))
            self.idx = [self.data.index[: len(self.train)], self.data.index[-len(test_data) : ]]

    def _prepare_folds(self, fold_strategy, fold):

        """ Assign the fold strategy"""

        self.logger.info("Assign the fold strategy.")

        if fold_strategy == "kfold":
            self.folds = KFold(n_splits = fold, random_state = self.seed, shuffle = True)
        elif fold_strategy == "stratified":
            self.folds = StratifiedKFold(n_splits = fold, random_state = self.seed, shuffle = True)
        else:
            raise ValueError(f"Invalid fold strategy: {fold_strategy}. Choose from 'kfold' or 'stratified'.")
        
    def _target_transformation(self, transformation_method):
        """Power transform the target data to be more Gaussian-like."""

        self.logger.info("Power transform the data to be more Gaussian-like.")

        if transformation_method == "yeo-johnson":
            transformation_estimator = PowerTransformer(method = "yeo-johnson", standardize = False, copy = False)

        elif transformation_method == "quantile":
            transformation_estimator = QuantileTransformer(output_distribution = "normal", copy = False, random_state = self.seed)

        else:
            raise ValueError(f"Invalid transformation method: {transformation_method}. Choose from 'yeo-johnson' or 'quantile'.")
        
        self.pipeline.steps.append(
            ("target_transformation", 
             TransformerWrapperWithInverse(
                 TargetTransformer(transformation_estimator)))
        
        )

    def _transformation(self, transformation_method):
        """Power transform the features to be more Gaussian-like."""

        self.logger.info("Power transform the features to be more Gaussian-like.")

        if transformation_method == "yeo-johnson":
            transformation_estimator = PowerTransformer(method = "yeo-johnson", standardize = False, copy = False)

        elif transformation_method == "quantile":
            transformation_estimator = QuantileTransformer(output_distribution = "normal", copy = False, random_state = self.seed)

        else:
            raise ValueError(f"Invalid transformation method: {transformation_method}. Choose from 'yeo-johnson' or 'quantile'.")

        self.pipeline.steps.append(("transformation", TransformerWrapper(transformation_estimator)))


    def _normalization(self, normalize_method):
        """Scale the features"""
        self.logger.into("Scale the features.")

        norm_dict = {
            "zscore" : StandardScaler(),
            "minmax" : MinMaxScaler()
        }

        if normalize_method in norm_dict:
            normalize_estimator = TransformerWrapper(norm_dict[normalize_method])

        else:
            raise ValueError(f"Invalid normalization method: {normalize_method}. Choose from {''.join(norm_dict)}.")

        self.pipeline.steps.append(("normalize", normalize_estimator))           

class RegressionModel(Preprocessor):
    
    def __init__(self):
        super().__init__()

    # Set up data =======================
    def setup(self, 
              data : Optional[DATAFRAME_LIKE] = None, 
              target : TARGET_LIKE = -1, 
              index : Union[bool, int, str, SEQUENCE_LIKE] = None, 
              test_size : float = 0.2,
              transformation : bool = False,
              transformation_method : str = "yeo-johnson",
              transform_target : bool = False,
              transform_target_method : str = "yeo-johnson",
              normalize : bool = False,
              normalize_method : str = "zscore",
              fold : int = 5,
              fold_strategy : Union[str, Any] = "kfold"):
        
        # Register the local parameters
        self._setup_params = {key :value for key, value in dict(locals()).items() if key != 'self' and value is not None}

        self.data = self._prepare_data(data = data, target = target)
        self.target_param = self.data.columns[-1]
        self.index = index

        self._prepare_folds(
            fold = fold, 
            fold_strategy = fold_strategy,
        )

        self._prepare_train_test(
            test_size = test_size,
            test_data = None
        )

        # Preprocessing =================== 

        self.pipeline = Pipeline(steps = [])

        if transformation:
            self._target_transformation(transformation_method)

        if normalize:
            self._normalization(normalize_method)

        if transformation:
            self._transformation(transformation_method)

        # Display ==========================
            
    def create_model(self,
                     estimator : Union[str, Any],
                     **kwargs):
        
        """
        estimator : str or scikit-learn compatible object

        * 'en' - Elastic Net
        * 'svm' - Support Vector Machine
        * 'ransac' - Random Sample Consensus
        * 'knn' - K Neighbors Regressor
        * 'rf' - Random Forest Regressor
        * 'xgboost' - Extreme Gradient Boosting
        * 'lightgbm' - Light Gradient Boosting Machine

        """
        

        self.logger.info("Importing Untrained Model")

        if isinstance(estimator, str) and estimator in AVAILABLE_ESTIMATORS.keys():
            model_definition = AVAILABLE_ESTIMATORS[estimator]['model']
            model_args = model_definition.args
            model_args = {**model_args, **kwargs}
            self.model_name = estimator
            self.model = model_definition(**model_args)
            self.pipeline.steps.append((self.model_name, self.model))

        

            

    def tune_model(self, 
                   estimator,
                   estimator_key : str,
                   fold : Optional[Union[int, Any]] = None,
                   round : int = 4,
                   n_iter : int = 10,
                   search_library : str = 'sckit-learn',
                   search_algorithm : Optional[str] = 'random',
                   early_stopping : Any = False,
                   early_stopping_max_iters : int = 10,
                   fit_kwargs : Optional[Dict] = None,
                   optimize : str = "MAPE",
                   **kwargs
                   ):
        
        # fold, groups
        function_params_str = ", ".join([f'{key} = {value}' for key, value in locals().items()])

        self.logger.info("Initialize the model tuning process.")
        self.logger.info(f"tune_model({function_params_str})")

        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        """
        
        Error Handling Starts Here

        """
            
        # Check for estimator
        if not hasattr(estimator, "fit"):
            raise ValueError(f"The estimator {estimator} does not have the required fit() method.")
        
        # Check for early_stopping parameter(Scheduler)
        possible_early_stopping = ["asha", "Hyperband", "Median"]
        if isinstance(early_stopping, str) and early_stopping not in possible_early_stopping:
            raise TypeError(f"early_stopping parameter must be one of {', '.join(possible_early_stopping)}")

        # Check for search_library
        possible_search_libraries = ["scikit-learn", "optuna", "scikit-optimize", "tune-sklearn"]
        if search_library not in possible_search_libraries:
            raise ValueError(f"search_library parameter must be one of {', '.join(possible_search_libraries)}")

        # Check for search_algorithm   
        if search_library == 'scikit-learn':

            if not search_algorithm:
                search_algorithm = 'random'

            possible_search_algorithm = ['random', 'grid']
            if search_algorithm not in possible_search_algorithm:
                raise ValueError(f"For 'scikit-learn' search_algorithm parameter must be one of {', '.join(possible_search_algorithm)}")
            
        """
        
        Error Handling Ends Here   
        
        """

        # Define the Parameter_Grid
        if search_library == "scikit-learn":
            
            # pipeline 단계에서 적용할 파라미터를 정의함
            
            if AVAILABLE_ESTIMATORS[estimator_key] is not None:
                search_kwargs = {**AVAILABLE_ESTIMATORS[estimator_key]['tune_args'], **kwargs}
            else:
                search_kwargs = {}

            if search_algorithm == "random":
                param_distributions = { self.model_name + '__' + key : value for key, value in AVAILABLE_ESTIMATORS[estimator_key]['tune_distributions'].items()}

                model_grid = RandomizedSearchCV(
                    estimator = self.pipeline,
                    param_distributions= param_distributions,
                    scoring = optimize,
                    n_iter = n_iter,
                    cv = fold,
                    random_state = self.seed,
                    n_jobs = -1,
                    refit = False,
                    **search_kwargs
                )
            
            else:
                
                param_grid = { self.model_name + '__' + key : value for key, value in AVAILABLE_ESTIMATORS[estimator_key]['tune_grid'].items()}

                model_grid = GridSearchCV(
                    estimator = self.pipeline,
                    param_grid = param_grid,
                    scoring = optimize,
                    cv = fold,
                    n_jobs = -1,
                    refit = False
                    **search_kwargs
                )

        def get_pipeline_fit_kwargs(pipeline : Pipeline, fit_kwargs : dict):
            try:
                model_step = pipeline.steps[-1]
            except Exception:
                return fit_kwargs
            
            if any(key.startswith(model_step[0]) for key in fit_kwargs.keys()):
                return fit_kwargs
            
            return {f"{model_step[0]}__{key}" : value for key, value in fit_kwargs.items()}
        
        fit_kwargs = get_pipeline_fit_kwargs(self.pipeline, fit_kwargs)
        
        # Implment the HyperParameter Tuning
        model_grid.fit(self.train.drop(columns = self.target_param), self.train[self.target_param], **fit_kwargs)

        best_params = model_grid.best_params_

        self.logger.info("HyperParameter Tuning is Completed")
        self.logger.info("Elapsed Time: {:.2f} seconds".format(time.time() - runtime_start))

        # Load the best model
        self.model = model_grid.best_estimator_
        CCC

if __name__ == '__main__':
    # Single Target은 그대로 적용 가능, Multi Target의 경우에는 2가지 방법을 생각해야함
    # 0. 우선 체크해야할 부분은 RandomForestRegressor, XgboostRegressor 등에 HyperParameter Tuning 적용했을 때 vs PyCaret의 방법을 적용했을 때 성능 차이가 심하게 발생하는 원인을 파악해야함.
    # 1. Multi Target을 각각의 독립적인 Target으로 고려해서 학습
    # 2. Multi Target을 Sequential 한 Target으로 고려해서 학습
    
    reg = RegressionModel()
    reg.setup(data = data, target = 'Settling_Time_2', index = False, test_size = 0.2)
    reg.create_model(estimator = 'xgboost')
    
    reg.tune_model(estimator = reg.model, estimator_key = reg.model_name, fold = 5, round = 4, n_iter = 10, search_library = 'scikit-learn', search_algorithm = 'random', early_stopping = False, early_stopping_max_iters = 10, fit_kwargs = None, optimize = 'MAPE')

    

    sys.exit(0)

    setup_results = setup(data = train , target='Settling_Time_2', normalize=True, verbose=True, log_experiment=True, experiment_name='data_comparison', system_log = os.path.join(DATA_DIR, 'logs\\data_comparison.log'))


    # 모델을 생성합니다

    # 모델을 튜닝합니다

    # 모델에 대한 그래프를 그립니다.