import os
import sys
import numpy as np
import glob
import pandas as pd
from function.calculate_frequency_quantity import get_lowcutoff_and_SNR
from utils.utils import logging_time, unit_from_name


import json
import pickle
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

SAVE_EXCEL_DIR = 'D:/001.Developement of Company Work/features'

if len(sys.argv) == 1:
    num_axes = 2
    interested_axis = list(map(int, '1,2'.split(',')))

else:
    num_axes = int(sys.argv[2])
    interested_axis = list(map(int, sys.argv[3].split(',')))

# 고정된 파라미터 값
xyz_param = 6
accel_param = 3

# 고정된 Column Index : 실험 방법을 바꾸거나 축의 개수/ 번호에 따라서 변화 가능.
def get_column_indices(num_axes, xyz_param, accel_param):
    column_indices = [[0],                                                                                  # time stamp            
                [9 + idx for idx in range(num_axes)],                                                       # cmd pls    9~11   
                [9 + num_axes + idx for idx in range(num_axes)],                                            # fdb pls    12~14
                None,                                                                                       # cmd rpm from pls None(15~17)  
                None,                                                                                       # fdb rpm from pls None(18~20)  
                [9 + 2*num_axes + idx for idx in range(xyz_param)],                                         # force sensor 15~20
                [9 + 2*num_axes + xyz_param + idx for idx in range(num_axes)],                         # torque  21~23
                [9 + 3*num_axes + xyz_param + idx for idx in range(num_axes)],                         # filter torque 24~26
                [9 + 8*num_axes + xyz_param + 2* accel_param + idx for idx in range(2)]           # ArmIO? 45~46
                ]
    return column_indices


# Outlier Condition 정의
TIME_INTERVAL = 0.35
OVERSHOOT_THRESHOLD = 1.5
VIBRATION_THRESHOLD = 30


def get_data_idx_from_features(column_indices):
    '''
    입력 파라미터 
        -. (column_indices)    PLS 데이터에서 추출하고자 하는 열의 인덱스들의 리스트,           type : [ [], [],... ,]
    실행 결과
        -. (data_indices_list) 데이터로부터 추출하고자 하는 열의 인덱스 시작점 관련 리스트,      type : []
                               (축의 개수에 따라 달라짐)
            -. 예상값 : [0, 1, 4, 7, 10, 13, 19, 22, 25, 28]
            -. 해석   : 시간값 ::  0번째 열
                        펄스 명령값 :: 1번째 열 (~ 3번째 열)
                        펄스 추종값 :: 4번째 열 (~ 6번째 열) 
                        속도 명령값 :: 7번째 열 (~ 9번째 열)
                        속도 추종값 :: 10번째 열 (~ 12번째 열)
                        힘 센서값   :: 13번째 열 (~ 18번째 열)
                        토크 추종값 :: 19번째 열 (~ 21번째 열)
                        필터 토크값 :: 22번째 열 (~ 24번째 열)
                        팔 IO 값   :: 25번째 열 (~ 27번째 열)
                        전체 column 개수에 대한 값
        -. (remained_indices)  원하는 축에 대한 관련 열의 인덱스 값들                   ,      type : []   
            -. 예상값 : [0, 1+2, 4+2, 7+2, 10+2, 13, 14, 15, 16, 17, 18, 19+2, 22+2, 25, 26]
            -. 해석   : column_indices 중 축의 번호와 관련 있는 값 중 원하는 축에 대한 인덱스 값(3,6,9,12,21,24) +
                        축의 번호와 무관한 값(0, 13, 14, 15, 16, 17, 18, 26)
    '''

    indices_list = [len(indices_in_list) if indices_in_list else num_axes for indices_in_list in column_indices]
    num_column_indices = sum(indices_list)
    data_indices_list = [sum(indices_list[:idx]) for idx, _ in enumerate(indices_list)] + [num_column_indices]

    if not interested_axis:
        raise ValueError
    
    else:
        # (예전 세팅)목표값 : 2,5,8,11,20,23
        # (현재 세팅)목표값 : 2,4,6,8,16,18 
        # (before) +1  ->  (now) +0
        #  
        remained_indices_dict = {'command_pulse' : [idx for idx in interested_axis], 
                 'feedback_pulse' : [num_axes + idx for idx in interested_axis],
                 'command_rpm' : [2 * num_axes + idx for idx in interested_axis],
                 'feedback_rpm' : [3 * num_axes + idx for idx in interested_axis],
                 'command_torque' : [6 + 4 * num_axes + idx for idx in interested_axis],
                 'feedback_torque' : [6 + 5 * num_axes + idx for idx in interested_axis]}

    return data_indices_list, remained_indices_dict


def column_extract_data(recv_data, data_indices_list, column_indices):
    '''
    원래 pd.Dataframe 데이터로 부터 필요한 column만 추출하여 np.array로 만드는 작업.
    입력 파라미터 :
        -. (recv_data)  생성된 데이터가 저장되어 있는 디렉토리                                type : pd.DataFrame
        -. (data_indices_list) 데이터로부터 추출하고자 하는 열의 인덱스 시작점 관련 리스트     type : []
        -. (column_indices)    원 데이터에서 추출하고자 하는 열의 인덱스들의 리스트            type : [ [], [],... ,]
    실행 결과 :
        -. data
    '''
    data = np.array([])
    status_list = ['command', 'feedback']
    status = 'command'
    for columns in column_indices:
        if data.shape[0] == 0:
            # 시간값 업데이트 & 스케일 및 원점 조절
            data = np.array(recv_data.iloc[:, columns])
            data = (data - data[0, columns])/1e6
        else:
            if columns:
                # 나머지값 업데이트
                data = np.concatenate((data, np.array(recv_data.iloc[:, columns])  ), 1)
         
            else:
                # 속도값 업데이트
                rpm_data = (data[ 1 :recv_data.shape[0], data_indices_list[status_list.index(status)+1] : data_indices_list[status_list.index(status)+2]] -data[ :recv_data.shape[0]-1, data_indices_list[status_list.index(status)+1] : data_indices_list[status_list.index(status)+2]] ) * 1000 * 60 /15360
                rpm_data = np.concatenate( (rpm_data, np.zeros((1,rpm_data.shape[1]))))
                data = np.concatenate( (data, rpm_data), 1  )
                status = 'feedback'      
    return data          

def row_extract_data(data, data_indices_list, Canny = False):
    '''
    입력 파라미터 : 
        -. (data) column 추출 이후 데이터                                       type : np.array([ [ ] ])
        -. (data_indices_list) 추출하고하는 열이 기입되어 있는 리스트             type : []
    실행 결과 : IO Extraction 통해서 불필요한 시간 데이터 삭제     
    '''
    IO_data = data[:,data_indices_list[-2]]
    IO_data[pd.isna(IO_data)] = '0x00000000'


    time_IO_index = [np.where(IO_data >'0x00000000')[0][0], np.where(IO_data >'0x00000000')[0][-1]]

    data = data[time_IO_index[0] : time_IO_index[1], :]

    return data, data

@logging_time
def test_one_from_directory_to_extract_raw_data(path_directories = [], Canny = False, motor_type = 'Hyeon_Seo', exp_num = 0,
                                                 num_axes = 2, column_indices = [[]]):
    '''
    Parameter Configuration이 기입되어 있는 1개의 (acc_time, real_inertia) 에 연관된 data_directory 에서 raw_data를 호출하여 신호의 특성값을 추출하고,
             이상치에 속하는지를 확인하고, raw_data에 대한 시각화를 완료한다.
    '''
    if not (np.array(interested_axis) <= num_axes).all():
        raise NotImplementedError
    
    data_indices_list, remained_indices = get_data_idx_from_features(column_indices)

    # info_dict : 각각의 실험 조건에 대한 특성값, 를 저장하는 dictionary
    info_dict = dict()

    for path in path_directories:
        if os.path.exists(os.path.join(path, 'raw_data.csv')):
            recv_data = pd.read_csv(os.path.join(path, 'raw_data.csv'), skipinitialspace = True)

            # data 중에서 필요한 column만 추출하기
            data = column_extract_data(recv_data, data_indices_list, column_indices)
            
            # data 중에서 유의미한 row(시간)만 추출하기 : IO Extraction
            reduced_cmd_data, reduced_fdb_data = row_extract_data(data, data_indices_list, Canny = False)

            # property 값, 이상치 값을 각각 properties.txt, classifying.txt 로 저장 및 raw_data.png 그리기
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                feature, quantity, outlier_status = summarize_raw_data(reduced_cmd_data, reduced_fdb_data, remained_indices, path)
        
                # dictionary 합치기
                information_one_condition = {**feature, **quantity, **outlier_status}
                condition = path.split('\\')[1]
                info_dict[condition] = information_one_condition
                print(condition, info_dict[condition])

    
    # 특성 dictionary를 특성 dataframe으로 변환 및 저장
    if os.path.exists(path_directories[0]):
        info_dataframe = pd.DataFrame(info_dict).T
        acc_time_real_inertia = "_".join(condition.split('_')[-4:])
        os.makedirs(os.path.join(SAVE_EXCEL_DIR, motor_type, acc_time_real_inertia, f'trial_{exp_num}' ), exist_ok = True)
        info_dataframe.to_excel(os.path.join(SAVE_EXCEL_DIR, motor_type, acc_time_real_inertia, f'trial_{exp_num}', 'dataframe.xlsx'))

    


@logging_time
def from_directory_to_extract_raw_data(data_directory, path_directories=[], Canny = False):
    '''
    Parameter Configuration이 기입되어 있는 data directory 에서 raw_data를 호출하여 신호의 특성값을 추출하고,
             이상치에 속하는지를 확인하고, raw_data에 대한 시각화를 완료한다.
    입력 파라미터
        -.(data_directory) Parameter Configuration가 표기된 디렉토리
        -.(Canny)          Canny Detection Algorithm을 추가적으로 수행하여 시간 간격으로 더 정확한 경계값을 얻을 것인지
    
    실행 결과 : 아래와 같은 구조로 raw_data 관련 파일들 생성
        - data
           - params_condition1
             - trial_num1
               -. raw_data.png
               -. properties.txt
               -. classifying.txt
        - dataframe(features & quantities & outlier_status)                        type : pd.DataFrame
    '''
    if not (np.array(interested_axis) <= num_axes).all():
        raise NotImplementedError

    
    data_indices_list, remained_indices = get_data_idx_from_features(column_indices)
    
    for path, _, file in os.walk(data_directory):
        if path not in path_directories:
            continue
       
        if file:
            # 데이터가 추출되지 않은 경우에만
            
            # if not os.path.exists(os.path.join(path, f'classifying_axis_{2}.txt')):
            #     print(path)
            # raw_data.csv로 부터 데이터 받기
            # string variable 띄어쓰기 삭제
            recv_data = pd.read_csv(os.path.join(path, 'raw_data.csv'), skipinitialspace = True)

            # data 중에서 필요한 column만 추출하기
            data = column_extract_data(recv_data, data_indices_list)
            # data 중에서 유의미한 row(시간)만 추출하기 : IO Extraction
            reduced_cmd_data, reduced_fdb_data = row_extract_data(data, data_indices_list, Canny = False)
            
            # property 값, 이상치 값을 각각 properties.txt, classifying.txt 로 저장 및 raw_data.png 그리기
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                feature, quantity, outlier_status = summarize_raw_data(reduced_cmd_data, reduced_fdb_data, remained_indices, path)

    # 실험에 사용할 데이터를 하나의 excel 파일로 저장하기

def summarize_raw_data(cmd_data, fdb_data, remained_list, save_dir, threshold = (0.1, 50)):
    '''
    Input : [Time/Pulse of Signal Data]                         type                    description                                 
            cmd_data                                            np.array()       (단위 응답함수) command 지령의 최대값 추출
            fdb_data                                            np.array()       feedback_pulse 시계열 값, 데이터 값 뽑아내기
            data_indices_list                                    []              데이터의 인덱스 뽑아내기
            save_dir                                             str             저장 장소
            threshold                                           (a,b)            진동 Peak 탐지 관련 하이퍼 파라미터
            
    Process : 0. 기본적인 신호의 특성값은 다음과 같이 계산한다.
                 - Start_Index = Time_idx_Fdb[0]
                 - Peak_Index = argmax(data) -> Peak_Time = time[, Peak_Index + Start_Index] - time[, Start_Index]
                 - Overshoot = data[, Peak_Index]
                 - Settling_Index = np.where( |data - max_cmd| > 0.02 * max_cmd ) -> Settling_Time = time[, Settling_Index + Start_Index] - time[, Start_Index]
                 - Rising_Index = ( np.where( data > 0.1 * max_cmd ), np.where( data  > 0.9 * max_cmd ) ) -> Rising_Time = time[, Rising_Index[1] + Start_Index] - time[, Rising_Index[0] + Start_Index]
                 - Delay_Index = np.where( data > 0.5 * max_cmd) -> Delay_Time = time[, Delay_Index + Start_Index]
              
              1. 입력된 파라미터 조합 중에서 데이터가 존재하는 경우(Label =1) 에서만 값을 갱신한다.
                 - 없으면 그대로 NaN 을 사용함
              2. 이상치에 대한 정의 및 처리
                 (정의)                                                                      (조건)
                 -1) Overshoot이 엄청 큰 경우                                                 Overshoot_Percentage >OVERSHOOT_THRESHOLD, Peak_Time < TIME_INTERVAL
                 -2) Overshoot이 크면서 수렴하지 않는 경우                                     1st_peak <= 2nd_peak     
                 -3) 반응 시간이 정의한 시간(TIME_INTERVAL)보다 길게 되는 경우 : 느린 반응성     Settling_Time > TIME_INTERVAL
                 -0) 이상치는 아니지만 처리가 필요한 경우 (OverDamped)                          Overshoot_Percentage < 1.0 in TIME_INTERVAL 
                
                     잔여진동이 많이 발생하는 경우에 대해서도 정의 및 처리가 필요할 것으로 예상(T.B.D)

                 (처리 순서)
                 -1) 첫번째 Peak <= 두번째 Peak : 발산, outlier_status(2)에 기록 (현재까지 실험에서는 주기적으로 감소하는 것을 확인했기 때문에)
                 -2) 모든 시간의 값을 TIME_INTERVAL 내로 들어오는지 체크함 
                     Settling_Time > TIME_INTERVAL인 경우 값을 NaN으로 반환 및 outlier_status(3)에 기록
                         Rise_Time > TIME_INTERVAL인 경우 값을 NaN으로 저장 
                         Delay_Time >TIME_INTERVAL인 경우 값을 NaN으로 저장
                     Peak_Time > TIME_INTERVAL, Overshoot_Percentage < 1.0 인 경우 Peak_Time, Overshoot_Percentage 는 마지막 시간 값 반환, outlier_status(4)에 기록
                                                                         
                     Peak_Time < TIME_INTERVAL, Overshoot_Percentage > OVERSHOOT_THRESHOLD 인 경우 Overshoot_Percentage NaN으로 반환, outlier_status(1)
                     
    Output :  
             [Characteristics of Signals]                       type            description                                                                 remarks
             Overshoot_Percentage                               float           Overshoot 발생시 가장 큰 peak 대비 cmd 지령의 비율
             Peak_Time                                          float           Overshoot 발생시 가장 큰 peak 까지의 시간
             Settling_time_(2)                                  float           feedback_pulse와 command_pulse 오차가 5,2% 이내로 들어갈때 까지 걸린 시간
             Delay_time                                         float           feedback_pulse가 command_pulse 50% 달성에 걸리는 시간
             Rise_time                                          float           feedback_pulse가 command_pulse 10%~90% 사이에 걸리는 시간
             Steady_State_Error                                 float           정상 상태 도달 이후 feedback_pulse와 command_pulse 사이 차이

             [Visualization of Signals]                                         각 특성값을 얻는데 걸리는 인덱스
             Settling_Time_2_idx                                float           
             Peak_Time_idx                                      float
             Delay_Time_idx                                     float
             Rise_Time_idx                                   [(start,end)]

             [Save the Characteristics of Outliers]
             outlier_status                                      dict            데이터의 이상 상태에 대한 저장
    
    '''
    # 원하는 축의 개수에 따라서 계산
    for index, axis_num in enumerate(interested_axis):
        # Command Pulse 불러오기, Max_CMD 에 대한 정의
        cmd_pulse = cmd_data[:, remained_list['command_pulse'][index]]
        max_cmd = np.max(cmd_pulse)
        cmd_rpm = cmd_data[:, remained_list['command_rpm'][index]]

        # Feedback Pulse 불러오기
        fdb_pulse = fdb_data[:, remained_list['feedback_pulse'][index]]
        fdb_time = fdb_data[:, 0]
        fdb_rpm = fdb_data[:, remained_list['feedback_rpm'][index]]
        fdb_torque = fdb_data[:, remained_list['feedback_torque'][index]]

        #print('Remained Indices : ', remained_list)
      
        # Feedback Pulse의 인덱스 크기, 시간    
        length_fdb_pulse = fdb_data.shape[0]
        time_interval = (fdb_data[-1,0] - fdb_data[0,0]) /2

        # 데이터 특성값 정의
        features = {'Overshoot':np.nan, 'Peak_Time':np.nan, 'Settling_Time_2':np.nan, 'Delay_Time':np.nan,
                    'Rise_Time':np.nan, 'Steady_State_Error':np.nan,
                    'Settling_Time_2_Idx' : np.nan, 'Peak_Time_Idx':np.nan, 'Delay_Time_Idx':np.nan, 'Rise_Time_Idx': (np.nan, np.nan)}
        quantities = {'pulse' : {'Dynamic_RMSE' : np.nan, 'Static_RMSE' : np.nan, 'Dynamic_MAE' : np.nan, 'Static_MAE' : np.nan},
                    'rpm' : {'Dynamic_RMSE' : np.nan, 'Static_RMSE' : np.nan, 'Dynamic_MAE' : np.nan, 'Static_MAE' : np.nan},
                    'torque' : {'Number_Frequency' : np.nan, 'SNR' : np.nan}}

        # Outliers 정의
        outlier_status = {'Unstable' : 0, 'Not_Converge' :0, 'Slow_Response' :0, 'No_Data':0, 'Vibration':0, 'Over_Damped': 0}

        # Torque Data 활용
        # 1. Number Of Frequency 계측
        middle_index = length_fdb_pulse//2
        dense,_ = find_peaks(fdb_torque[:middle_index].flatten(), prominence = threshold[0])
        sparse,_ = find_peaks(fdb_torque[:middle_index].flatten(), prominence = threshold[1], height = 200)

        peak_indices, _ = find_peaks(fdb_pulse[ : length_fdb_pulse //2 ].flatten(), prominence = 10, height = 20) # 400 *0.04 = 16
        sorted_peak_indices = sorted(peak_indices, key = lambda idx: fdb_pulse[idx], reverse = True)
        
        quantities['torque']['Number_Frequency'] = int(len(dense) - len(sparse))
        
        # 2. SNR 계산
        
        '''
        입력값 :    - torque data
                   - idx_parameter : I.R_0050_D.R_0067_F.P_0010_S.P_0200_I.P_0200_A.T_0.100_R.I_0009
                   - save_dir : cutoff frequency 구한 후에 그래프를 저장할 장소 
                                D:/001.Developement of Company Work/data/AutoTuning\\I.R_0.050_D.R_0067_F.P_0010_S.P_0200_I.P_0200_A.T_0.010_R.I_0000\\Hyeon_Seo\\trial_0 
                   - show_graph : cutoff frequency 구한 후에 그래프를 보여주는지에 대한 여부
        실행 결과 : 
        '''
        _, snr = get_lowcutoff_and_SNR(fdb_torque, save_dir.split('\\')[1], save_dir, show_graph = True)
        quantities['torque']['SNR'] = snr


        # Outlier Condition 정의
        Slow_Response = time_interval > TIME_INTERVAL
        Not_Converge = len(sorted_peak_indices) >=2 and sorted_peak_indices[0] >= sorted_peak_indices[1] and np.max( fdb_pulse[: length_fdb_pulse //2]/max_cmd) > 1 
        Unstable = time_interval <=TIME_INTERVAL and np.max(fdb_pulse[:length_fdb_pulse //2] /max_cmd) > OVERSHOOT_THRESHOLD
        Converge_OverDamped = time_interval <= TIME_INTERVAL and np.where( (np.abs(fdb_pulse[: length_fdb_pulse //2] - max_cmd) >= 0.02 * max_cmd)  )[0].shape[0] == 0
        Frequently_Vibration = quantities['torque']['Number_Frequency'] > VIBRATION_THRESHOLD

        # Settling에 도달하지 못한 경우
        if Slow_Response:
            outlier_status['Slow_Response'] = 1
            features['settling_Time_2_Idx'] = length_fdb_pulse//2
            features['Peak_Time_Idx'] = length_fdb_pulse//2
            
            # delay time 이 정해진 시간(TIME_INTERVAL) 보다 더 늦어지는 경우에 대한 분류도 필요함.
            if np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.5 * max_cmd))[0].shape[0] == 0:
                # delay_time >TIME_INTERVAL 인 경우
                features['Delay_Time_Idx'] = length_fdb_pulse//2
            else:
                features['Delay_Time_Idx'] = int( np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.5 * max_cmd))[0][0])
            
            # rise time 이 정해진 시간(TIME_INTERVAL) 보다 더 늦어지는 경우에 대한 분류도 필요함.
            if np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0].shape[0] == 0:
                # left interval values of rising time(10%) > TIME_INTERVAL
                features['Rise_Time_Idx'] = (int(length_fdb_pulse//2)-1, int(length_fdb_pulse//2)  )
            elif np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.9 * max_cmd))[0].shape[0] == 0 and \
                  np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0].shape[0] != 0:
                # right interval values of rising time(90%) > TIME_INTERVAL, left(10%) < TIME_INTERVAL
                features['Rise_Time_Idx'] = (int (np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0][0] ), int(length_fdb_pulse//2)  )
            else:
                features['Rise_Time_Idx'] = ( int (np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0][0] ),
                                        int ( np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.9 * max_cmd))[0][0] ) )
            
            # 경우에 따라서 threshold(50) 다르게 해야할 수 도 있다.
            
            if Frequently_Vibration:
                outlier_status['Vibration'] = 1
        
        
        # 발산하는 경우(첫번째 피크 < 두번째 피크)
        if Not_Converge:
            outlier_status['Not_Converge'] = 1
            features['Settling_Time_2_Idx'] = length_fdb_pulse//2
            features['Peak_Time_Idx'] = int(sorted_peak_indices[0])
            features['Delay_Time_Idx'] = int( np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.5 * max_cmd))[0][0])
            features['Rise_Time_Idx'] = ( int (np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0][0] ),
                                        int ( np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.9 * max_cmd))[0][0] ) )
            
            # 경우에 따라서 threshold(50) 다르게 해야할 수 도 있다.
            quantities['torque']['Number_Frequency'] = int(len(dense) - len(sparse))
            
            if Frequently_Vibration:
                outlier_status['Vibration'] = 1

        # 수렴하긴 하지만(첫번째 피크 > 두번째 피크) Overshoot이 상당히 큰 경우 :
        if Unstable:
            outlier_status['Unstable'] = 1
            features['Peak_Time_Idx'] = int(np.argmax(fdb_pulse[: length_fdb_pulse //2]))
            features['Settling_Time_2_Idx'] = int(np.where( (np.abs(fdb_pulse[ features['Peak_Time_Idx']: length_fdb_pulse//2] - max_cmd) >= 0.02 *max_cmd  ) )[0][-1]   ) +\
                                                features['Peak_Time_Idx']
            features['Delay_Time_Idx']  = int( np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.5 * max_cmd))[0][0])
            features['Rise_Time_Idx'] = ( int (np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0][0] ),
                                        int ( np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.9 * max_cmd))[0][0] ) )
            
            features['Overshoot'] = float(np.max(fdb_pulse[: length_fdb_pulse//2])/max_cmd)
            features['Peak_Time'] = float(fdb_time[features['Peak_Time_Idx']] - fdb_time[0]  )
            features['Settling_Time_2'] = float(fdb_time[features['Settling_Time_2_Idx']] - fdb_time[0])
            features['Delay_Time'] = float(fdb_time[features['Delay_Time_Idx']] - fdb_time[0])
            features['Rise_Time'] = float(fdb_time[features['Rise_Time_Idx'][1]] - fdb_time[features['Rise_Time_Idx'][0]] )
            features['Steady_State_Error'] = float( np.abs( fdb_pulse[features['Settling_Time_2_Idx']] - max_cmd)   )

            for data_type in ['pulse', 'rpm']:
                # feedback signal 
                
                quantities[data_type]['Dynamic_RMSE'] =  float(np.sqrt(np.mean(np.square(locals()['fdb_'+data_type][: features['Settling_Time_2_Idx']] - locals()['cmd_'+data_type][: features['Settling_Time_2_Idx']]     ))) ) 
                quantities[data_type]['Static_RMSE'] = float(np.sqrt(np.mean(np.square(locals()['fdb_'+data_type][: middle_index] - locals()['cmd_'+data_type][: middle_index]       )))) 
                quantities[data_type]['Dynamic_MAE'] = float(np.mean(np.abs(locals()['fdb_'+data_type][: features['Settling_Time_2_Idx']  ] - locals()['cmd_'+data_type][:features['Settling_Time_2_Idx']]  ) ) ) 
                quantities[data_type]['Static_MAE'] = float(np.mean(np.abs(locals()['fdb_'+data_type][: middle_index] - locals()['cmd_'+data_type][: middle_index])) ) 


            # 경우에 따라서 threshold(50) 다르게 해야할 수 도 있다.
            if Frequently_Vibration:
                outlier_status['Vibration'] = 1

        # 수렴하긴 하지만 Overdamped 상태인 경우
        if Converge_OverDamped:
            outlier_status['Over_Damped'] = 1
            features['Settling_Time_2_Idx'] = length_fdb_pulse//2
            features['Peak_Time_Idx'] = length_fdb_pulse//2

            # delay time 이 정해진 시간(0.35) 보다 더 늦어지는 경우에 대한 분류도 필요함.
            if np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.5 * max_cmd))[0].shape[0] == 0:
                # delay_time >0.35 인 경우
                features['Delay_Time_Idx'] = length_fdb_pulse//2
            else:
                features['Delay_Time_Idx'] = int( np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.5 * max_cmd))[0][0])
            
            # rise time 이 정해진 시간(0.35) 보다 더 늦어지는 경우에 대한 분류도 필요함.
            if np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0].shape[0] == 0:
                # left interval values of rising time(10%) > 0.35
                features['Rise_Time_Idx'] = (int(length_fdb_pulse//2)-1, int(length_fdb_pulse//2)  )
            elif np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.9 * max_cmd))[0].shape[0] == 0 and \
                  np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0].shape[0] != 0:
                # right interval values of rising time(90%) > 0.35, left(10%) < 0.35
                features['Rise_Time_Idx'] = (int (np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0][0] ), int(length_fdb_pulse//2)  )
            else:
                features['Rise_Time_Idx'] = ( int (np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0][0] ),
                                        int ( np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.9 * max_cmd))[0][0] ) )

        
            features['Overshoot'] = float(np.max(fdb_pulse[: length_fdb_pulse//2])/max_cmd)
            features['Peak_Time'] = float(fdb_time[features['Peak_Time_Idx']] - fdb_time[0]  )
            features['Settling_Time_2'] = float(fdb_time[features['Settling_Time_2_Idx']] - fdb_time[0])
            features['Delay_Time'] = float(fdb_time[features['Delay_Time_Idx']] - fdb_time[0])
            features['Rise_Time'] = float(fdb_time[features['Rise_Time_Idx'][1]] - fdb_time[features['Rise_Time_Idx'][0]] )
            features['Steady_State_Error'] = float( np.abs( fdb_pulse[features['Settling_Time_2_Idx']] - max_cmd)   )

            for data_type in ['pulse', 'rpm']:
                # feedback signal 
                
                quantities[data_type]['Dynamic_RMSE'] =  float(np.sqrt(np.mean(np.square(locals()['fdb_'+data_type][: features['settling_time_2_idx']] - locals()['cmd_'+data_type][: features['settling_time_2_idx']]     ))) ) 
                quantities[data_type]['Static_RMSE'] = float(np.sqrt(np.mean(np.square(locals()['fdb_'+data_type][: middle_index] - locals()['cmd_'+data_type][: middle_index]       )))) 
                quantities[data_type]['Dynamic_MAE'] = float(np.mean(np.abs(locals()['fdb_'+data_type][: features['settling_time_2_idx']  ] - locals()['cmd_'+data_type][:features['settling_time_2_idx']]  ) ) ) 
                quantities[data_type]['Static_MAE'] = float(np.mean(np.abs(locals()['fdb_'+data_type][: middle_index] - locals()['cmd_'+data_type][: middle_index])) )

            # 경우에 따라서 threshold(50) 다르게 해야할 수 도 있다.
            if Frequently_Vibration:
                outlier_status['Vibration'] = 1

        # 그 외 나머지 정상적인 경우
        if (not Slow_Response) and (not Not_Converge) and (not Unstable) and (not Converge_OverDamped):
            features['Peak_Time_Idx'] = int(np.argmax(fdb_pulse[: length_fdb_pulse //2]))
            
            # delay time 이 정해진 시간(TIME_INTERVAL) 보다 더 늦어지는 경우에 대한 분류도 필요함.
            if np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.5 * max_cmd))[0].shape[0] == 0:
                # delay_time >TIME_INTERVAL 인 경우
                features['Delay_Time_Idx'] = length_fdb_pulse//2
            else:
                features['Delay_Time_Idx'] = int( np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.5 * max_cmd))[0][0])
            
            # rise time 이 정해진 시간(TIME_INTERVAL) 보다 더 늦어지는 경우에 대한 분류도 필요함.
            if np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0].shape[0] == 0:
                # left interval values of rising time(10%) > TIME_INTERVAL
                features['Rise_Time_Idx'] = (int(length_fdb_pulse//2)-1, int(length_fdb_pulse//2)  )
            elif np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.9 * max_cmd))[0].shape[0] == 0 and \
                  np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0].shape[0] != 0:
                # right interval values of rising time(90%) > TIME_INTERVAL, left(10%) < TIME_INTERVAL
                features['Rise_Time_Idx'] = (int (np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0][0] ), int(length_fdb_pulse//2)  )
            else:
                features['Rise_Time_Idx'] = ( int (np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.1 * max_cmd))[0][0] ),
                                        int ( np.where(   (np.abs(fdb_pulse[ : length_fdb_pulse //2 ]) >= 0.9 * max_cmd))[0][0] ) )
            
            features['Settling_Time_2_Idx'] = int(np.where( (np.abs(fdb_pulse[  np.min((features['Delay_Time_Idx'] ,features['Peak_Time_Idx'])): length_fdb_pulse//2] - max_cmd) >= 0.02 *max_cmd  ) )[0][-1] ) + \
                                                int(np.min((features['Delay_Time_Idx'] ,features['Peak_Time_Idx'])))
            
            features['Overshoot'] = float(np.max(fdb_pulse[: length_fdb_pulse//2])/max_cmd)
            features['Peak_Time'] = float(fdb_time[features['Peak_Time_Idx']] - fdb_time[0]  )
            features['Settling_Time_2'] = float(fdb_time[features['Settling_Time_2_Idx']] - fdb_time[0])
            features['Delay_Time'] = float(fdb_time[features['Delay_Time_Idx']] - fdb_time[0])
            features['Rise_Time'] = float(fdb_time[features['Rise_Time_Idx'][1]] - fdb_time[features['Rise_Time_Idx'][0]] )
            features['Steady_State_Error'] = float( np.abs( fdb_pulse[features['Settling_Time_2_Idx']] - max_cmd)   )

            for data_type in ['pulse', 'rpm']:
                # feedback signal 
                
                quantities[data_type]['Dynamic_RMSE'] =  float(np.sqrt(np.mean(np.square(locals()['fdb_'+data_type][: features['settling_time_2_idx']] - locals()['cmd_'+data_type][: features['settling_time_2_idx']]     ))) ) 
                quantities[data_type]['Static_RMSE'] = float(np.sqrt(np.mean(np.square(locals()['fdb_'+data_type][: middle_index] - locals()['cmd_'+data_type][: middle_index]       )))) 
                quantities[data_type]['Dynamic_MAE'] = float(np.mean(np.abs(locals()['fdb_'+data_type][: features['settling_time_2_idx']  ] - locals()['cmd_'+data_type][:features['settling_time_2_idx']]  ) ) ) 
                quantities[data_type]['Static_MAE'] = float(np.mean(np.abs(locals()['fdb_'+data_type][: middle_index] - locals()['cmd_'+data_type][: middle_index])) ) 


            # 경우에 따라서 threshold(50) 다르게 해야할 수 도 있다.
            if Frequently_Vibration:
                outlier_status['Vibration'] = 1


        save_data(features, os.path.join(save_dir, f'properties_axis_{axis_num}.txt'))
        save_data(quantities, os.path.join(save_dir, f'performance_axis_{axis_num}.txt'))
        save_data(outlier_status, os.path.join(save_dir, f'classifying_axis_{axis_num}.txt'))
        
        plot_raw_data(cmd_data, fdb_data, remained_list, save_dir, features)

        return features, quantities, outlier_status

        
def save_data(values, directory):
    with open(directory, "w") as file:
        file.write(json.dumps(values))

def plot_raw_data(cmd_data, fdb_data, remained_list, save_dir, features): 
    plt.rcParams["figure.figsize"] = (16,12)
    plt.rcParams["axes.grid"] = True

    fig = plt.figure(clear = True)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    plt.subplots_adjust(hspace = 0.35)
  
    for index, axis_num in enumerate(interested_axis):
        # Command Pulse 불러오기, Max_CMD 에 대한 정의
        cmd_pulse = cmd_data[:, remained_list['command_pulse'][index]]
        cmd_time = cmd_data[:, 0]
        cmd_rpm = cmd_data[:, remained_list['command_rpm'][index]]
        cmd_torque = cmd_data[:, remained_list['command_torque'][index]]

        # Feedback Pulse 불러오기
        fdb_pulse = fdb_data[:, remained_list['feedback_pulse'][index]]
        fdb_time = fdb_data[:, 0]
        fdb_rpm = fdb_data[:, remained_list['feedback_rpm'][index]]
        fdb_torque = fdb_data[:, remained_list['feedback_torque'][index]]
        title_string_from_dir = save_dir.split('\\')[-3]
        print("What is the save_dir? ",save_dir)
        title_string = ",".join([ name + '=' + value + unit_from_name(name) for name, value in zip(title_string_from_dir.split('_')[::2], title_string_from_dir.split('_')[1::2]) ])
        
        ax1.set_title(  title_string  , fontsize = 20)

        ax1.plot(cmd_time, cmd_pulse, label='cmd_{}'.format(interested_axis[index]))   # cmd
        ax1.plot(fdb_time, fdb_pulse, label='fb_{}'.format(interested_axis[index]))    # fdb
        ax1.plot(fdb_time, cmd_pulse-fdb_pulse, label='err_{}'.format(interested_axis[index]))   # err
        ax1.set_xticks([   fdb_time[features['Settling_Time_2_Idx']], fdb_time[features['Peak_Time_Idx']], fdb_time[features['Delay_Time_Idx']] ] )
                #np.arange(data[num, time_idx[num][0],0], data[num, time_idx[num][0],1], 0.1).tolist() +
        ax1.text(x = np.round(fdb_time[features['Settling_Time_2_Idx']],3), y = -10 , s = r'$t_s$' ,ha = 'center')
        ax1.text(x = np.round(fdb_time[features['Peak_Time_Idx']],3), y = -60, s = r'$t_p$', ha = 'center')
        ax1.text(x = np.round(fdb_time[features['Delay_Time_Idx']],3), y = -110, s = r'$t_d$', ha = 'center')
        ax1.annotate(r'$t_r$', xytext= (fdb_time[features['Rise_Time_Idx'][0]], 0 ), xy = (fdb_time[features['Rise_Time_Idx'][1]], 0), xycoords = 'data',
                        arrowprops = dict(arrowstyle = '<->', color = 'red', lw =2) )
        #ax1.axvline(x = settling_time_2_idx[num], color ='r', linestyle = '-', linewidth = 1, label = r'$t_s$')

        
        ax1.legend()
        ax1.set_xlabel('time(s)', fontsize = 14)
        ax1.set_ylabel('Magnitue of pulse', fontsize = 14)

        ax2.set_title("rpm(1:cmd,2:fb)")
        ax2.plot( cmd_time, cmd_rpm, label='cmdrpm_{}'.format(interested_axis[index]))   # rpm(based on cmd)
        ax2.plot( fdb_time, fdb_rpm, label='fbrpm_{}'.format(interested_axis[index]))   # rpm(based on fdb)
        ax2.legend()
        ax2.set_xlabel('time(s)', fontsize =14)
        ax2.set_ylabel('Manitude of RPM(rad/s)', fontsize = 14)

        ax3.set_title("torque(1:torque,2:filter_torque)")
    
        ax3.plot( cmd_time,cmd_torque, label='torq_{}'.format(interested_axis[index]))   # cmd
        ax3.plot( fdb_time,fdb_torque, label='filtorq_{}'.format(interested_axis[index]))   # filter
        ax3.legend()
        ax3.set_xlabel('time(s)', fontsize = 14)
        ax3.set_ylabel(r'Manitude of Torque(pulse/$s^2$)', fontsize = 14)
        plt.savefig(save_dir+f'/raw_data_axis_{axis_num}.png', dpi = 200)

        plt.close()



if __name__ =='__main__':

    # 1. 데이터를 불러와서 특성값을 추출하여 저장, 원본 데이터를 시각화 하여 저장

    data_directory = 'D:/001.Developement of Company Work/data/AutoTuning'

    if len(sys.argv) == 1:
        exp_num = 0
        exp_condition = 'meta_learning_task'
        
    else:
        exp_num = int(sys.argv[1])
        exp_condition = sys.argv[4]#sys.argv[3]


    config_dir = './config/remote_config.json'
    with open(config_dir, "r") as file:
        exp_params = json.load(file)[exp_condition]

    name_list = ['inertia_gain', 'damper_gain', 'first_pole', 'second_pole', 'integral_pole', 'acc_time', 'real_inertia']
    for params_name in name_list:
        globals()[params_name+'_list'] = list(map(int if not params_name == 'acc_time' else float,  exp_params[0][params_name].split(',') ) )
    
    if len(acc_time_list) != 1 and len(real_inertia_list) != 1:
        raise ValueError('The length of acc_time_list and real_inertia_list should be 1 in this version!!')
    
    
    acc_time = acc_time_list[0]
    real_inertia = real_inertia_list[0]

    '''
    (데이터 정리) 실험 방법에 따라서 관심축, 전체 축의 개수, column_indices를 설정
    '''
    interested_axes = dict()
    num_axes_dict = dict()
    column_indice_dict = dict()
    if exp_condition == 'meta_learning_task':
        num_axes_dict['Small_Joint'] = num_axes
        num_axes_dict['Zero_460W'] = num_axes
        if real_inertia == 0 :
            num_axes_dict['Hyeon_Seo'] = num_axes
            interested_axes['Hyeon_Seo'] = [2]
        else:
            num_axes_dict['Hyeon_Seo'] = 3
            interested_axes['Hyeon_Seo'] = [3]
        
        interested_axes['Small_Joint'] = [1]
        interested_axes['Zero_460W'] = [2]
    
    elif exp_condition in ['finner_experiment_Zero_460W']:
        num_axes_dict[ "_".join(exp_condition.split('_')[-2:]) ] = num_axes
        interested_axes[ "_".join(exp_condition.split('_')[-2:]) ] = [2]

    elif exp_condition in ['finer_experiment_Small_Joint']: 
        num_axes_dict[ "_".join(exp_condition.split('_')[-2:]) ] = num_axes
        interested_axes[ "_".join(exp_condition.split('_')[-2:]) ] = [1]

    elif exp_condition in ['finer_experiment_Hyeon_Seo']:
        if real_inertia == 0:
            num_axes_dict[ "_".join(exp_condition.split('_')[-2:]) ] = num_axes
        else:
            num_axes_dict[ "_".join(exp_condition.split('_')[-2:]) ] = 3
        interested_axes[ "_".join(exp_condition.split('_')[-2:]) ] = [2]

    for key in num_axes_dict.keys():
        column_indice_dict[key] = get_column_indices(num_axes_dict[key], xyz_param, accel_param)

    # path directories 정의
    path_directories = {key:[] for key in interested_axes.keys()}
    for inertia_gain in inertia_gain_list:
        for damper_gain in damper_gain_list:
            for first_pole in first_pole_list:
                for second_pole in second_pole_list:
                    for integral_pole in integral_pole_list:
                        for acc_time in acc_time_list:
                            for real_inertia in real_inertia_list:
                                
                                for keys in interested_axes.keys(): 
                                    File_Path = f'D:/001.Developement of Company Work/data/AutoTuning\\I.R_{int(inertia_gain):04d}_D.R_{int(damper_gain):04d}_F.P_{int(first_pole):04d}_S.P_{int(second_pole):04d}_I.P_{int(integral_pole):04d}_A.T_{float(acc_time):.3f}_R.I_{int(real_inertia):04d}\\{keys}\\trial_{exp_num}'
                                    path_directories[keys].append(File_Path)         
    
    print(path_directories)
    print('Hello!')
 
    if exp_condition == 'meta_learning_task':
        
        for key, value in interested_axes.items():
            interested_axis = value
            num_axes = num_axes_dict[key]
            column_indices = column_indice_dict[key]
            test_one_from_directory_to_extract_raw_data(path_directories = path_directories[key], 
                                                        motor_type = key, 
                                                        exp_num = exp_num,
                                                        num_axes = num_axes,
                                                        column_indices = column_indices 
                                                        )
            #from_directory_to_extract_raw_data(data_directory, path_directories=path_directories)
    else:
        #from_directory_to_extract_raw_data(data_directory, path_directories=path_directories)
        test_one_from_directory_to_extract_raw_data(path_directories = path_directories[list(path_directories.keys())[0]],
                                                    motor_type = list(path_directories.keys())[0], 
                                                    exp_num = exp_num,
                                                    num_axes = num_axes_dict[list(path_directories.keys())[0]],
                                                    column_indices = column_indice_dict[list(path_directories.keys())[0]]
                                                    )


    # 2. 제로 모터에 사용한 클러스터링 결과를 저장한 이후 소관절 모터에 사용한 Classifying 적용
    load_cluster = False

    if load_cluster:
        pass

    else:
        pass

    
