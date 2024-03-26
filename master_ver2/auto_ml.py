import sys
import numpy as np
import pandas as pd
import json
import os

from sklearn.model_selection import train_test_split
from pycaret.regression import *


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

# Data Preprocessing
data = pd.read_excel(f'{DATA_DIR}\\dataframe.xlsx', index_col = 0)

# train - test split    
train, test = train_test_split(data, test_size = 0.2, random_state = 42)


# PyCaret 환경을 설정합니다.
os.makedirs(os.path.join(DATA_DIR, 'logs'), exist_ok=True)  
reg = setup(data = train , target= 'Settling_Time_2', normalize=True, verbose=True, log_experiment=True, experiment_name='motor_experiment', system_log = os.path.join(DATA_DIR, 'logs\\motor_experiment.log'))



# 여러 모델을 비교합니다.
# best_model = compare_models(sort='MAPE')  # MAPE 기준으로 정렬하여 최적의 모델을 찾습니다.
# print("What is the best model?", best_model)

# 하나의 모델을 생성합니다.
best_model = create_model('xgboost')  # XGBoost 모델을 생성합니다.

# 특정 모델을 생성하고 튜닝합니다.
tuned_model = tune_model(best_model, optimize='MAPE', verbose=True, tuner_verbose = 4)  # MAPE를 최소화하도록 하이퍼파라미터를 조정합니다.
print("Tuned Model: ", tuned_model)


import sys
sys.exit(0)

# 최종 모델을 확정합니다.
final_model = finalize_model(tuned_model)
print("Final Model: ", final_model)

# 새로운 데이터에 대해 예측을 수행합니다.
predictions = predict_model(final_model, data=test)
print("Predictions: ", predictions) 

# 모델을 그립니다.
#plot_model(final_model, plot = 'residuals_interactive')
plot_model(final_model, plot='residuals')
plot_model(final_model, plot='cooks') 
plot_model(final_model, plot='rfe')
plot_model(final_model, plot='feature')
plot_model(final_model, plot='feature_all')
plot_model(final_model, plot ='error')
plot_model(final_model, plot ='vc')
plot_model(final_model, plot='manifold') # t-SNE plot
plot_model(final_model, plot='learning')
plot_model(final_model, plot='parameter')
# plot_model(final_model, plot='tree') 
# Decision Tree plot is only available for sckikt-learn Decision Trees and Forests,
# Ensemble models using those or Stacked models using those as meta final estimators.

# 모델을 평가합니다
evaluate_model(final_model)

# 모델을 해석합니다 : binary classification 만 가능
# interpret_model(final_model)
