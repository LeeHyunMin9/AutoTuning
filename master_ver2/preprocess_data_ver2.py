import os
import sys
import numpy as np
import glob
import pandas as pd
from utils.utils import logging_time, unit_from_name
import json
import pickle
import matplotlib.pyplot as plt

# HyperParameters of System Communication
if len(sys.argv) == 1:
    num_axes = 2
    interested_axis = list(map(int, '1,2'.split(',')))

else:
    num_axes = int(sys.argv[2])
    interested_axis = list(map(int, sys.argv[3].split(',')))

# 고정된 파라미터 값
xyz_param = 6
accel_param = 3

####### parameter order in excel #########

# int_data    info->ticks/1000                                   0
# int_data    gRBSTS.manipower.lastval                           1
# int_data    gRBSTS.svsts                                       2
# bit_datas    info->_sysio                                      3
# bit_datas    gRBSTS.holdinfo.flags                             4
# int_data    gRBSTS.last_hold_factor                            5
# int_data    gRBSTS.vrbdesc[0].sts                              6
# int_data    gRBSTS.vrbdesc[1].sts                              7
# int_data    gRBSTS.n_queued                                    8
# int_data    for (i = 1:n_axes) : info->pcmdp[i]             9~11
# int_data    for (i = 1:n_axes) : info->pcrup[i]            12~14
# int_data    for (i = 1:force_sensor_axes) : gRBSTS.force_sensor_sts.value[i] 15~20
# int_data    for (i = 1:n_axes) : (info->torque[i]<<16)>>16  21~23
# int_data    for (i = 1:n_axes) : (info->_fltdQ[i]<<16)>>16  24~26
# int_data    gSHMO->rbsts.cmdx                                27
# int_data    gSHMO->rbsts.cmdy                                28
# int_data    gSHMO->rbsts.cmdz                                29
# int_data    gSHMO->rbsts.cmdrz                               30
# int_data    gSHMO->rbsts.cmdry                               31
# int_data    gSHMO->rbsts.cmdrx                               32
# bit_datas    for (i = 1:n_axes) : info->_402ctrl[i]          33~35
# bit_datas    for (i = 1:n_axes) : info->_402sts[i]           36~38
# bit_datas    for (i = 1:n_axes) : info->_402err[i]           39~41
# bit_datas    for (i = 1:n_axes) : info->_sts[i]              42~44
# bit_datas    for (i = 1:n_axes) : info->_stdio[i]            45~47
# bit_datas    for (i = 1:n_axes) : info->_armio[i]            48~50
# int_data    info->_accx                                      51
# int_data    info->_accy                                      52
# int_data    info->_accz                                      53
# int_data    for (i = 1:n_axes) : info->btmp[i]

class DataCollector:
    # 1. 데이터를 수집하는 클래스
    def __init__(self, num_axes, interested_axes, save_dir):
        self.num_axes = num_axes
        self.interested_axes = interested_axes  
        self.save_dir = save_dir
        self.define_dataAdr(num_axes)
        self.define_colorAdr(num_axes)

    def define_dataAdr(self, num_axes):

        self.array_integer_Adr_ = {                       # When 6 Axis
            "time_stamp_Adr":0,                     #      "time_stamp_Adr":    0,
            "manip_power_Adr":1,                    #     "manip_power_Adr":    1,
            "svsts_Adr":2,                          #           "svsts_Adr":    2,
            "verbsts_Adr":3,                        #         "verbsts_Adr":    3,
            "cmdpls_Adr":7+num_axes*0,              #          "cmdpls_Adr":    7,  (7 + 4*0 )
            "fbpls_Adr":7+num_axes*1,               #           "fbpls_Adr":   11,  (7 + 4*1 )
            "force_sensor_Adr":7+num_axes*2,        #    "force_sensor_Adr":   15,  (7 + 4*2 )  <-- 6
            "torque_Adr":7+num_axes*2+6,            #          "torque_Adr":   21,  (7 + 4*2 + 6)
            "filter_torque_Adr": 7+num_axes*3+6,    #   "filter_torque_Adr":   25,  (7 + 4*3 + 6)
            "xyz_Adr":7+num_axes*4+6 ,              #             "xyz_Adr":   29,  (7 + 4*4 + 6)  <-- 6
            "accl_Adr":7+num_axes*4+6*2,            #            "accl_Adr":   35,  (7 + 4*4 + 6*2 )  <-- 3
            "tempAdr":7+num_axes*4+6*2+3 ,          #             "tempAdr":   38,  (7 + 4*4 + 6*2 + 3) # 0x523e
            "thermal_torque_Adr":7+num_axes*5+6*2+3,#  "thermal_torque_Adr":   42,  (7 + 4*5 + 6*2 + 3) # 0x523d
            "volage_Adr":7+num_axes*6+6*2+3 ,       #          "volage_Adr":   46,  (7 + 4*6 + 6*2 + 3) # 0x523f
            "board_tempAdr":7+num_axes*7+6*2+3 ,    #       "board_tempAdr":   50,  (7 + 4*7 + 6*2 + 3) # 0x523e
            "target_pos_Adr":7+num_axes*8+6*2+3 ,   #      "target_pos_Adr":   54,  (7 + 4*8 + 6*2 + 3) # 0x5232
            "402_actpos_Adr":7+num_axes*9+6*2+3 ,   #      "402_actpos_Adr":   58,  (7 + 4*8 + 6*2 + 3) # 0x6064
            "402_flwerr_Adr":7+num_axes*10+6*2+3 ,  #      "402_flwerr_Adr":   62,  (7 + 4*8 + 6*2 + 3) # 0x60f4
            }
        
        self.array_string_Adr_ ={
            "time_stamp_Adr":"time_stamp",
            "manip_power_Adr":"manip_power",
            "svsts_Adr":"svsts",
            "verbsts_Adr":"verbsts",
            "cmdpls_Adr":"cmdpls",
            "fbpls_Adr":"fbpls",
            "force_sensor_Adr":"force_sensor",
            "torque_Adr":"torque",
            "filter_torque_Adr":"filter_torque",
            "xyz_Adr":"xyz",
            "accl_Adr":"accl",
            "tempAdr":"temp",
            "thermal_torque_Adr":"thermal_torque",
            "volage_Adr":"volage",
            "board_tempAdr":"board_temp",
            "target_pos_Adr":"target_pos",
            "402_actpos_Adr":"402_actpos",
            "402_flwerr_Adr":"402_flwerr",
        }

        self.last_integer_Adr_index_ = num_axes
        self.N_int_columns = max(self.array_integer_Adr_.values()) + self.last_integer_Adr_index_

        self.bit_array_Adr_ = {
            "system_io"         : 0,
            "holdinfo_flags"    : 1,
            "_402ctrl"          : 2,
            "_402err"           : 2 + num_axes*1,
            "_sts"              : 2 + num_axes*2,
            "_stdio"            : 2 + num_axes*3,
            "_armio"            : 2 + num_axes*3 + 1   
        }

        self.last_bit_Adr_index_ = 1

        self.N_columns = max(self.array_integer_Adr_.values()) + self.last_integer_Adr_index_ + max(self.bit_array_Adr_.values())+ self.last_bit_Adr_index_
        '''
        bit_index = [3, 
                    4, 
                    9 + num_axes*4 + xyz_param * 2,
                    9 + num_axes*5 + xyz_param * 2, 
                    9 + num_axes*6 + xyz_param * 2,
                    9 + num_axes*7 + xyz_param * 2,
                    9 + num_axes*8 + xyz_param * 2,
                    9 + num_axes*9 + xyz_param * 2,]
        '''
        #self.column_names = [ if col not in bit_index else for col in range(self.N_columns)]


    def define_colorAdr(self, num_axes):
        self.colorMap = ["peru","orangered","orange","yellow","limegreen","deepskyblue","violet","gray","white"]
        self.legendMap = ["j{}".format(i+1) for i in range(num_axes)]


    @logging_time
    def test_one_from_directory_to_extract_raw_data(self, path_directories = []):
        '''
        Parameter Configuration이 기입되어 있는 1개의 data_directory 에서 raw_data를 호출하여 신호의 특성값을 추출하고,
                이상치에 속하는지를 확인하고, raw_data에 대한 시각화를 완료한다.
        '''
        if not (np.array(self.interested_axes) <= self.num_axes).all():
            raise NotImplementedError("The interested axes are not in the range of num_axes")
         
        extract_column_names = ['time_stamp','cmdpls', 'fbpls', 'force_sensor', 'torque', 'filter_torque', 'armio']
        for path in path_directories:
            recv_data = pd.read_csv(os.path.join(path, 'raw_data.csv'), skipinitialspace= True)
            XX
if __name__ == '__main__':

    exp_num = 0
    exp_condition = 'dummy_finner_experiment'
    data_directory = 'D:/001.Developement of Company Work/data/AutoTuning'
    keys = 'Hyeon_Seo'

    config_dir = './config/remote_config.json'
    with open(config_dir, "r") as file:
        exp_params = json.load(file)[exp_condition]

    name_list = ['inertia_gain', 'damper_gain', 'first_pole', 'second_pole', 'integral_pole', 'acc_time', 'real_inertia']
    for params_name in name_list:
        globals()[params_name+'_list'] = list(map(int if not params_name == 'acc_time' else float,  exp_params[0][params_name].split(',') ) )

    path_directories = []
    for inertia_gain in inertia_gain_list:
        for damper_gain in damper_gain_list:
            for first_pole in first_pole_list:
                for second_pole in second_pole_list:
                    for integral_pole in integral_pole_list:
                        for acc_time in acc_time_list:
                            for real_inertia in real_inertia_list: 
                                File_Path = f'D:/001.Developement of Company Work/data/AutoTuning\\I.R_{int(inertia_gain):04d}_D.R_{int(damper_gain):04d}_F.P_{int(first_pole):04d}_S.P_{int(second_pole):04d}_I.P_{int(integral_pole):04d}_A.T_{float(acc_time):.3f}_R.I_{int(real_inertia):04d}\\{keys}\\trial_{exp_num}'
                                path_directories.append(File_Path)
    
    
    data = DataCollector(num_axes, interested_axis, save_dir = data_directory)
    data.test_one_from_directory_to_extract_raw_data(path_directories)
    