import os
import sys
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV   
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pycaret.regression import *

from utils.utils import logging_time
from plot.utils import metric, PredictionErrorDisPlayDistances
from function.load_model import Individual_RegressionModel


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

# Save Numpy Array to JSON File
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)

# Data Preprocessing
class DataPreparation:

    def __init__(self, data_dir, preprocessing_method = None):
        self.data_dir = data_dir
        self.preprocessing_method = preprocessing_method
        data = self.load()
        self.split_data(data)

    def load(self):
        data = pd.read_excel(f'{self.data_dir}\\dataframe.xlsx', index_col = 0)
        return data
    
    def split_data(self, data, features = ['Settling_Time_2'], split_ratio = 0.2):
        # Extract interesting features
        self.features = features
        target_data = data[features]
        target_data = self.preprocessing_data(target_data)

        # Split Index and Value 
        # Index : I.R_0050_D.R_0067_F.P_0025_S.P_0200_I.P_0200_A.T_0.010_R.I_0009 -> [50.0, 67.0, 25.0, 200.0, 200.0, 0.01, 9.0]
        # Value : Feature Value
        # index = np.array([ list(map(float, idx.split('_')[1::2])) for idx in target_data.index])
        # value = target_data.values

        # Split Train and Test Data
        self.train_index, self.test_index, self.y_train, self.y_test = train_test_split(target_data.index, target_data.values , test_size = split_ratio, random_state = 42)

        self.X_train = np.array([ list(map(float, idx.split('_')[1::2])) for idx in self.train_index])
        self.X_test = np.array([ list(map(float, idx.split('_')[1::2])) for idx in self.test_index])
        

    def preprocessing_data(self, target_data):
        # Preprocessing NAN values
        if self.preprocessing_method == None:
            return target_data

        else:
            return target_data.fillna(0)

# Accuracy of Regression Model
class RegressionModel:

    def __init__(self, model_type = 'RandomForestRegressor',
                 params_dir = DATA_DIR, results_dir = DATA_DIR):
        
        # Data Preparation
        data = DataPreparation(DATA_DIR)
        self.features = data.features
        self.X_train = data.X_train
        self.X_test = data.X_test
        self.y_train = data.y_train
        self.y_test = data.y_test

        self.train_index = data.train_index
        self.test_index = data.test_index

        self.model_type = model_type
        self.params_dir = os.path.join(params_dir, model_type)
        os.makedirs(self.params_dir, exist_ok = True)
        self.results_dir = os.path.join(results_dir, model_type)
        os.makedirs(self.results_dir, exist_ok = True)

    @logging_time
    def fit(self):
        if self.model_type == 'RandomForestRegressor':
            
            self.model = RandomForestRegressor(random_state = 42)

            # Scaler
            self.scaler = MinMaxScaler()
            self.y_train = self.scaler.fit_transform(self.y_train)
            
            if isinstance(self.scaler, MinMaxScaler):
                self.scaler_params = {'min' : self.scaler.data_min_, 'max' : self.scaler.data_max_}
            elif isinstance(self.scaler, StandardScaler):
                self.scaler_params = {'mean' : self.scaler.mean_, 'var' : self.scaler.var_}

            self.model.fit(self.X_train, self.y_train)
            self.model_params = self.model.get_params()
            # Save the parameters and results
            self.save_params()
            self.save_results()

        elif self.model_type == 'XGBoost':
            # Initial Setting
            self.final_model = XGBRegressor(learning_rate = 0.1, n_estimators = 1000, seed = 42)

            # Scaler
            # 이상치 외곡
            self.scaler = StandardScaler()

            # Make Pipeline
            self.model = make_pipeline(self.scaler, self.final_model)

            # HyperParameter Tuning
            self.model = tune_model(self.model, optimize='MAPE', verbose=True, tuner_verbose = 4)
            #self.hyperparameter_tuning(search_algorithm = 'random')

            if isinstance(self.scaler, MinMaxScaler):
                self.scaler_params = {'min' : self.scaler.data_min_, 'max' : self.scaler.data_max_}
            elif isinstance(self.scaler, StandardScaler):
                self.scaler_params = {'mean' : self.scaler.mean_, 'var' : self.scaler.var_}
            
            
            #best_params = self.hyperparameter_tuning()   
            
            #self.model.fit(self.X_train, self.y_train)

            self.model_params = self.model.get_params()
            # Save the parameters and results
            #self.save_params()
            self.save_results()

        elif self.model_type == 'LightGBM':
            pass

        elif self.model_type == 'SVM':
            
            self.model = SVR(C = 1.0, epsilon = 0.2)
            # C : Regularization Parameter, epsilon : margin of tolerance
            
            # Scaler
            self.scaler = MinMaxScaler()
            self.y_train = self.scaler.fit_transform(self.y_train)

            if isinstance(self.scaler, MinMaxScaler):
                self.scaler_params = {'min' : self.scaler.data_min_, 'max' : self.scaler.data_max_}
            elif isinstance(self.scaler, StandardScaler):
                self.scaler_params = {'mean' : self.scaler.mean_, 'var' : self.scaler.var_}

            
            best_params = self.hyperparameter_tuning()
            self.model.set_params(**best_params)

            self.model.fit(self.X_train, self.y_train)
            self.model_params = self.model.get_params()
            
            # Save the parameters and results
            self.save_params()
            self.save_results()

        elif self.model_type == 'AutoKeras':
            pass

        elif self.model_type == 'TorchNN':
            
            # model
            self.model = Individual_RegressionModel()

            if not os.path.exists(f'{self.params_dir}/checkpoints'):
                # Training

                # dataset
                X_train_tensor = torch.tensor(self.X_train, dtype = torch.float32)
                Y_train_tensor = torch.tensor(self.y_train, dtype = torch.float32)
                X_val_tensor = torch.tensor(self.X_test, dtype = torch.float32)
                Y_val_tensor = torch.tensor(self.y_test, dtype = torch.float32)

                # Normalization
                # min_train_X, max_train_X = torch.min(X_train_tensor, axis = 0).values, torch.max(X_train_tensor, axis =0).values
                # min_train_Y, max_train_Y = torch.min(Y_train_tensor, axis = 0).values, torch.max(Y_train_tensor, axis =0).values

                # DataLoader 생성 - batch_size를 설정해주지 않아서 기본값인 1로 설정됨 : BatchNorm1d가 정상적으로 작동 x
                # 1->32로 수정
                train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
                
                val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
                val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
                
                # trainer callback
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    dirpath = f'{self.params_dir}/checkpoints',
                    filename = 'best-checkpoint-{epoch:02d}-{val_loss:.2f}',
                    monitor = 'val_loss',
                    mode = 'min',
                    save_top_k = 5
                )

                # trainer
                trainer = pl.Trainer(callbacks=[checkpoint_callback], 
                                    max_epochs = 100, logger = WandbLogger(name = 'Individual_RegressionModel', project = 'Individual_RegressionModel'))

                trainer.fit(self.model, train_loader, val_loader)

            else:
                # Load the trained model
                checkpoints = os.listdir(f'{self.params_dir}/checkpoints')
                # val_loss, epoch 작은 순서대로 정렬 : best-checkpoint-epoch=00-val_loss=0.00.ckpt
                str_compiler = re.compile(r'epoch=(\d+)-val_loss=(\d+.\d+)')
                checkpoints.sort( key = lambda x : (float(str_compiler.search(x).group(2)), int(str_compiler.search(x).group(1))) )

                self.model = Individual_RegressionModel.load_from_checkpoint(f'{self.params_dir}/checkpoints/{checkpoints[0]}')
                self.model.eval()

            self.save_results()

    @logging_time
    def hyperparameter_tuning(self, search_algorithm = 'random'):

        if self.model_type != 'XGBoost':
            if self.model_type == 'RandomForestRegressor':
                param_grid = {
                    'n_estimators' : [100, 200, 300, 400, 500],
                    'max_features' : ['auto', 'sqrt', 'log2'],
                    'max_depth' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                    'min_samples_split' : [2, 5, 10],
                    'min_samples_leaf' : [1, 2, 4],
                    'bootstrap' : [True, False]
                }

            elif self.model_type == 'SVM':
                param_grid = {
                    'C' : [0.1, 1, 10, 100],
                    'gamma' : ['scale', 1, 0.1, 0.01, 0.001],
                    'kernel' : ['rbf', 'sigmoid']
                }

            grid = GridSearchCV(self.model, param_grid, cv = 5, scoring = ['neg_mean_absolute_error', 'neg_mean_absolute_percentage_error'], return_train_score = True, refit = 'neg_mean_absolute_error', verbose = 10)
            grid.fit(self.X_train, self.y_train)
            print(f'Best Parameters : {grid.best_params_}')
            print(f'Best Estimator : {grid.best_estimator_}')
            print(f'Best Score : {grid.best_score_}')

            return grid.best_params_
        
        elif self.model_type == 'XGBoost':
            # n_estimators : number of boosting rounds
            # learning_rate : step size shrinkage used to prevent overfitting
            # gamma : minimum loss reduction required to make a further partition on a leaf node of the tree
            # alpha : L1 regularization term on weights
            # lambda : L2 regularization term on weights
            # scale_pos_weight : control the balance of positive and negative weights, useful for unbalanced classes
            # subsample : the fraction of samples to be used for fitting the individual base learners
            # colsample_bytree : the fraction of features to be used for fitting the individual base learners
            
            # [Sequential] HyperParameter Tuning Method
            # 1. Fix Learning Rate and Number of Estimators
            # 2. Tuning Max Depth and Min Child Weight
            # 3. Tuning Gamma
            # 4. Tuning Subsample and Colsample Bytree
            # 5. Tuning Regularization Parameters
            # 6. Reducing Learning Rate and Number of Estimators
            # 7. Ensemble with seeds
            
            param_grid = {'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1,0.5], 
                        'n_estimators': [10, 50, 100, 150, 300],
                        'subsample': [0.2, 0.5, 0.7, 1], 
                        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
                        'colsample_bytree': [0.5, 0.7, 0.9, 1], 
                        'min_child_weight': [1, 2, 3, 4], 
                        'reg_alpha': [0, 0.01, 0.1, 1, 5, 10], 
                        'reg_lambda': [0,  0.01, 0.1, 1, 5, 10], 
                        'scale_pos_weight': [i for i in np.arange(0, 50, 10)]}

            # change param_grid key : self.model.steps[-1][0] + '__' + key (ex. 'xgbregressor__n_estimators')
            mapping = {key : self.model.steps[-1][0] + '__' + key for key in param_grid.keys()} 
            param_grid = {mapping[key] : value for key, value in param_grid.items()}


            if search_algorithm == 'random':
                # PipeLine with Model
                grid = RandomizedSearchCV(estimator = self.model, 
                                          param_distributions = param_grid, 
                                          cv = 5, 
                                          scoring = 'neg_mean_absolute_percentage_error', 
                                          return_train_score = True, 
                                          refit = False, 
                                          random_state = 42,
                                          n_iter = 100,
                                          verbose = 10)

            elif search_algorithm == 'grid':  
                grid = GridSearchCV(estimator = self.model,
                                    param_grid = param_grid,
                                    cv = 5,
                                    scoring = 'neg_mean_absolute_percentage_error', 
                                    return_train_score = True, 
                                    refit = False, 
                                    random_state = 42,
                                    n_iter = 100,
                                    verbose = 10)        
            
            grid.fit(self.X_train, self.y_train)
            best_params = grid.best_params_
            best_score = grid.best_score_
            print(f'Best Score : {best_score}')
            # Seq2_param_grid = {
            #     'max_depth' : [3, 6, 9],
            #     'min_child_weight' : [1, 3, 5]
            # }
            # grid_Seq2 = GridSearchCV(self.model, Seq2_param_grid, cv = 5, scoring = ['neg_mean_absolute_error', 'neg_mean_absolute_percentage_error'], return_train_score = True, refit = 'neg_mean_absolute_error', verbose = 10)
            # grid_Seq2.fit(self.X_train, self.y_train)
            # self.model.set_params(**grid_Seq2.best_params_)

            # Seq3_param_grid = {
            #     'gamma' : [0, 0.1, 0.2, 0.3, 0.4]
            # }
            # grid_Seq3 = GridSearchCV(self.model, Seq3_param_grid, cv = 5, scoring = ['neg_mean_absolute_error', 'neg_mean_absolute_percentage_error'], return_train_score = True, refit = 'neg_mean_absolute_error', verbose = 10)
            # grid_Seq3.fit(self.X_train, self.y_train)
            # self.model.set_params(**grid_Seq3.best_params_)

            # Seq4_param_grid = {
            #     'subsample' : [0.6, 0.7, 0.8, 0.9, 1.0],
            #     'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0]
            # }
            # grid_Seq4 = GridSearchCV(self.model, Seq4_param_grid, cv = 5, scoring = ['neg_mean_absolute_error', 'neg_mean_absolute_percentage_error'], return_train_score = True, refit = 'neg_mean_absolute_error', verbose = 10)
            # grid_Seq4.fit(self.X_train, self.y_train)
            # self.model.set_params(**grid_Seq4.best_params_)

            # Seq5_param_grid = {
            #     'reg_alpha' : [0, 1e-5, 0.01, 0.1, 1, 10],
            #     'reg_lambda' : [0, 1e-5, 0.01, 0.1, 1, 10]
            # }
            # grid_Seq5 = GridSearchCV(self.model, Seq5_param_grid, cv = 5, scoring = ['neg_mean_absolute_error', 'neg_mean_absolute_percentage_error'], return_train_score = True, refit = 'neg_mean_absolute_error', verbose = 10)
            # grid_Seq5.fit(self.X_train, self.y_train)
            # self.model.set_params(**grid_Seq5.best_params_)

            # Seq6_param_grid = {
            #     'learning_rate' : [0.1, 0.05, 0.01],
            #     'n_estimators' : [100, 1000, 5000]
            # }
            # grid_Seq6 = GridSearchCV(self.model, Seq6_param_grid, cv = 5, scoring = ['neg_mean_absolute_error', 'neg_mean_absolute_percentage_error'], return_train_score = True, refit = 'neg_mean_absolute_error', verbose = 10)
            # grid_Seq6.fit(self.X_train, self.y_train)
            # self.model.set_params(**grid_Seq6.best_params_)
            
            # best_params = {**grid_Seq2.best_params_, **grid_Seq3.best_params_, **grid_Seq4.best_params_, **grid_Seq5.best_params_, **grid_Seq6.best_params_}
            
            return best_params
            
       

    def predict(self):
        pass

    def load_model(self):
        pass

    def save_params(self):
        if self.scaler_params:
            # Save the scaler parameters json file
            with open(f'{self.params_dir}/scaler_params.json', 'w') as f:
                json.dump(self.scaler_params, f, cls = NumpyEncoder)

        if self.transformer_params:
            # Save the transformer parameters json file
            with open(f'{self.params_dir}/transformer_params.json', 'w') as f:
                json.dump(self.transformer_params, f, cls = NumpyEncoder)

        # Save the model parameters
        with open(f'{self.params_dir}/model_params.json', 'w') as f:
            json.dump(self.model_params, f)
            
    def save_results(self):
        # Save the validation individual quality with excel file
        if self.model_type in ['RandomForestRegressor', 'XGBoost' ,'SVM']:
            # transformer 가 PowerTransformer 이고, scaler 가 StandardScaler 일 때
            if isinstance(self.transformer, PowerTransformer):
                # transformer -> scaler 순서로 역변환 해줘야함.
                
                y_valid_before_transform = self.model.predict(self.X_train)
                y_true_train_before_transform = self.y_train
                y_pred_before_transform = self.model.predict(self.X_test)

                if y_valid_before_transform.ndim == 1:
                    y_valid_before_transform = y_valid_before_transform.reshape(-1, 1)
                    y_true_train_before_transform = y_true_train_before_transform.reshape(-1, 1)
                    y_pred_before_transform = y_pred_before_transform.reshape(-1, 1)
                
                y_valid = self.transformer.inverse_transform(y_valid_before_transform)
                y_true_train = self.transformer.inverse_transform(y_true_train_before_transform)

                # Save the prediction individual quality with excel file    
                y_pred = self.transformer.inverse_transform(y_pred_before_transform)

                # mean, variance = self.scaler_params['mean'], self.scaler_params['var']
                # y_valid = (y_valid) * np.sqrt(variance) + mean
                # y_true_train = (y_true_train) * np.sqrt(variance) + mean

                # y_pred = (y_pred) * np.sqrt(variance) + mean

                
            elif isinstance(self.transformer, PowerTransformer) and isinstance(self.scaler, MinMaxScaler):
                # transformer -> scaler 순서로 역변환 해줘야함.
                y_valid = self.model.predict(self.X_train)
                y_true_train = self.y_train
                y_pred = self.model.predict(self.X_test)

                

            # self.scaler 가 MinMaxScaler 일 때
            elif not isinstance(self.transformer, PowerTransformer) and isinstance(self.scaler, MinMaxScaler):
                min, max = self.scaler_params['min'], self.scaler_params['max']
                y_valid = self.model.predict(self.X_train) * (max - min) + min
                y_true_train = self.y_train * (max - min) + min

                # Save the prediction individual quality with excel file
                y_pred = self.model.predict(self.X_test) * (max - min) + min
            
            # self.scaler 가 StandardScaler 일 때
            elif not isinstance(self.transformer, PowerTransformer) and isinstance(self.scaler, StandardScaler):
                mean, variance = self.scaler_params['mean'], self.scaler_params['var']
                y_valid = self.model.predict(self.X_train)
                y_true_train = self.y_train

                # Save the prediction individual quality with excel file
                y_pred = self.model.predict(self.X_test)  + mean

            else:
                y_valid = self.model.predict(self.X_train)#np.expm1(self.model.predict(self.X_train))
                y_true_train = self.y_train#np.expm1(self.y_train)
                y_pred = self.model.predict(self.X_test)#np.expm1(self.model.predict(    np.log1p(self.X_test)      ))

            

        elif self.model_type == 'TorchNN':
            # self.X_train 
            y_valid = self.model(torch.tensor(self.X_train, dtype = torch.float32)).detach().numpy()
            y_true_train = self.y_train

            y_pred = self.model(torch.tensor(self.X_test, dtype = torch.float32)).detach().numpy()

        y_true_test = self.y_test  

        if y_valid.ndim == 1:
            y_valid = y_valid.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)

        validation_dataframe = pd.DataFrame()
        validation_dataframe.index = self.train_index
        prediction_dataframe = pd.DataFrame()
        prediction_dataframe.index = self.test_index
        for target_object in self.features:
            validation_dataframe['True_' + target_object] = y_true_train[:, self.features.index(target_object)]
            prediction_dataframe['True_' + target_object] = y_true_test[:, self.features.index(target_object)]
            
            for measure in ['MAE', 'MAPE', 'MSLE']:
                validation_dataframe[f'{measure}_' + target_object] = metric(y_true_train[:, self.features.index(target_object)],
                                                                                  y_valid[:, self.features.index(target_object)], type = measure) 
                prediction_dataframe[f'{measure}_' + target_object] = metric(y_true_test[:, self.features.index(target_object)],
                                                                                  y_pred[:, self.features.index(target_object)], type = measure)    

        validation_dataframe.to_excel(f'{self.results_dir}/validation_quality.xlsx')
        prediction_dataframe.to_excel(f'{self.results_dir}/prediction_quality.xlsx')

        # Save the prediction quality with json file
        summary_dict = dict()
        for measure in ['MAE', 'MAPE', 'MSLE']:
            for target_object in self.features:
                summary_dict[f'{measure}_{target_object}_mean'] = { 'train_mean' : validation_dataframe[f"{measure}_{target_object}"].mean(), 'test_mean' : prediction_dataframe[f"{measure}_{target_object}"].mean() }

                summary_dict[f'{measure}_{target_object}_var'] = { 'train_var' : validation_dataframe[f"{measure}_{target_object}"].var(), 'test_var' : prediction_dataframe[f"{measure}_{target_object}"].var() }
       
        with open(f'{self.results_dir}/validation_quality.json', 'w') as f:
            json.dump(summary_dict, f, cls = NumpyEncoder)

        # with open(f'{self.results_dir}/prediction_quality.json', 'w') as f:
        #     json.dump(prediction_dataframe, f, cls= NumpyEncoder)


        # Save the individual prediction quality with png file(histogram)
        for target_object in self.features:

            for measure in ['MAE', 'MAPE', 'MSLE']:

                if not os.path.exists(f'{self.results_dir}/{measure}'):
                    os.makedirs(f'{self.results_dir}/{measure}', exist_ok = True)

                fig, ax = plt.subplots(1, 2, figsize = (10, 5))
                ax[0].hist( metric(y_true_train[:, self.features.index(target_object)], y_valid[:, self.features.index(target_object)], measure), bins = 50, alpha = 0.5, label = 'Validation')
                ax[0].set_title(f'Validation Quality of {target_object}')
                ax[0].legend()
                ax[1].hist( metric(y_true_test[:, self.features.index(target_object)], y_pred[:, self.features.index(target_object)], measure), bins = 50, alpha = 0.5, label = 'Prediction')
                ax[1].set_title(f'Prediction Quality of {target_object}')
                ax[1].legend()
                plt.savefig(f'{self.results_dir}/{measure}/{target_object}_quality.png')
                plt.close()

                # Save the prediction error with png file
                fig, ax = plt.subplots(2, 2, figsize = (20, 15))
                # subplot 간의 간격 자동 조절
                plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
                
                ax[0,0].set_title(f'Validation Quality of {target_object}')
                PredictionErrorDisPlayDistances(y_true = y_true_train[:, self.features.index(target_object)], y_pred = y_valid[:,self.  
                features.index(target_object)], distance = measure).plot(ax = ax[0,0], kind = 'actual_vs_predicted')
                
                ax[0,1].set_title(f'Validation Quality of {target_object}')
                PredictionErrorDisPlayDistances(y_true = y_true_train[:, self.features.index(target_object)], y_pred = y_valid[:,self.  
                features.index(target_object)], distance = measure).plot(ax = ax[0,1])

                ax[1,0].set_title(f'Prediction Quality of {target_object}')
                PredictionErrorDisPlayDistances(y_true = y_true_test[:, self.features.index(target_object)], y_pred = y_pred[:,self.  
                features.index(target_object)], distance = measure).plot(ax = ax[1,0], kind = 'actual_vs_predicted', scatter_kwargs={'color':'orange'})

                ax[1,1].set_title(f'Prediction Quality of {target_object}')
                PredictionErrorDisPlayDistances(y_true = y_true_test[:, self.features.index(target_object)], y_pred =  y_pred[:, self.features.index(target_object)], distance = measure).plot(ax = ax[1,1], scatter_kwargs={'color':'orange'})

                
                plt.savefig(f'{self.results_dir}/{measure}/{target_object}_prediction_error.png')
                plt.close()


        


    # Grid Search
    # param_grid = {
    #     'n_estimators' : [100, 200, 300, 400, 500],
    #     'max_features' : ['auto', 'sqrt', 'log2'],
    #     'max_depth' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #     'min_samples_split' : [2, 5, 10],
    #     'min_samples_leaf' : [1, 2, 4],
    #     'bootstrap' : [True, False]
    # }
    # grid_search = GridSearchCV(rf, param_grid, cv = 5, scoring = mse_scorer, return_train_score = True)
    
    # import time
    # start_time = time.time()
    # grid_search.fit(X_train, y_train)
    # end_time = time.time()
    # print(f'Working Time : {end_time - start_time :.3f} sec')
    # print(f'Best Parameters : {grid_search.best_params_}')
    # print(f'Best Estimator : {grid_search.best_estimator_}')
    # print(f'Best Score : {grid_search.best_score_}')



if __name__ == '__main__':

    model = RegressionModel(model_type = 'XGBoost')
    model.fit()
    
    # find_approximated_model(index, value, test_size = 0.2)   