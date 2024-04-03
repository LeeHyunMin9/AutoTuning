import sys
import numpy as np
import pandas as pd
import json
import os

from sklearn.metrics import r2_score, mean_absolute_percentage_error
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
data = data[['Settling_Time_2']]
motor_params = pd.DataFrame(np.array([ list(map(float, idx.split('_')[1::2])) for idx in data.index]), columns = ['Inertia_Ratio', 'Damping_Ratio', 'First_Pole', 'Second_Pole', 'Integral_Pole', 'Acc_Time', 'Real_Inertia'], index = data.index)
data = pd.concat([motor_params, data], axis = 1)


# train - test split    
train, test = train_test_split(data, test_size = 0.2, random_state = 42)




if __name__ == '__main__':

    
            
        
    # PyCaret 환경을 설정합니다.
    os.makedirs(os.path.join(DATA_DIR, 'logs'), exist_ok=True)  
    reg = setup(data = train , target= 'Settling_Time_2', normalize=True, transform_target=True, transform_target_method= 'yeo-johnson', 
        
                transformation = True, transformation_method = 'yeo-johnson', verbose=True, log_experiment='wandb', experiment_name='motor_experiment', system_log = os.path.join(DATA_DIR, 'logs\\motor_experiment.log'))



    # 여러 모델을 비교합니다.
    # best_model = compare_models(sort='MAPE')  # MAPE 기준으로 정렬하여 최적의 모델을 찾습니다.
    # print("What is the best model?", best_model)

    # Seed 별 xgboost 모델을 생성합니다.

    best_model = create_model('xgboost')  # XGBoost 모델을 생성합니다.

    # 특정 모델을 생성하고 튜닝합니다.
    tuned_model = tune_model(best_model, optimize='MAPE', verbose=True, tuner_verbose = 1, n_iter = 50)  # MAPE를 최소화하도록 하이퍼파라미터를 조정합니다.
    print("Tuned Model: ", tuned_model)




    # 최종 모델을 확정합니다.
    final_model = finalize_model(tuned_model)
    print("Final Model: ", final_model)

    # 새로운 데이터에 대해 예측을 수행합니다.
    predictions = predict_model(final_model, data=test)
    print("Predictions: ", predictions) 

    # Prediction score
    print(final_model.score(test.drop(columns = 'Settling_Time_2'), test['Settling_Time_2'] ))
    print(final_model.score(predictions.drop(columns = ['Settling_Time_2', 'prediction_label']), predictions[['Settling_Time_2', 'prediction_label']] ))

    # 기준(r2 > 0.9, mape < 0.01)을 만족하는 모델을 저장합니다.

    y_true = np.array(predictions['Settling_Time_2']).reshape(-1, 1)
    y_pred = np.array(predictions['prediction_label']).reshape(-1, 1)

    # nan prediction 처리
    y_true = y_true[~np.isnan(y_pred)]
    y_pred = y_pred[~np.isnan(y_pred)]

    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print("R2: ", r2)
    print("MAPE: ", mape)



    # 모델을 그립니다.
    #plot_model(final_model, plot = 'residuals_interactive')
    plot_model(final_model, plot='residuals')
    #plot_model(final_model, plot='cooks') 
    #plot_model(final_model, plot='rfe')
    #plot_model(final_model, plot='feature')
    #plot_model(final_model, plot='feature_all')
    plot_model(final_model, plot ='error')
    XXXX
    #plot_model(final_model, plot ='vc')
    #plot_model(final_model, plot='manifold') # t-SNE plot
    #plot_model(final_model, plot='learning')
    #plot_model(final_model, plot='parameter')
    # plot_model(final_model, plot='tree') 
    # Decision Tree plot is only available for sckikt-learn Decision Trees and Forests,
    # Ensemble models using those or Stacked models using those as meta final estimators.

    # 모델을 평가합니다
    #evaluate_model(final_model)

    # 모델을 해석합니다 : binary classification 만 가능
    # interpret_model(final_model)

    if r2 > 0.9 and mape < 0.05:
        
        save_model(final_model, os.path.join(DATA_DIR, 'final_model'))
        print("Model is saved.")


    else:
        print("Model is not saved.")

        

   