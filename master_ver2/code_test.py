#!/usr/bin/python
#-*- coding: utf-8 -*-

import time
import subprocess
import datetime
import os
import rblib
import sys
import json
import glob
from i611_io import *

def svstat(rb):
    res = rb.ioctrl(128, 0, 0xffffffff, 0, 0xffffffff)
    if res[0] == False:
        print("Unable to read servo power")
        raise Exception
    if (res[1] & 0x01)!=0:
        return 1
    return 0

def cal_sleep_time(number_of_files):
    '''
    1개당 500KB
    Subprocess 생성 시간 + 실행 시간 + Waiting Time + File Transfer Time
    Cubic Regression Model : 0.06x^3 + 5.32x^2 - 6.35x + 1.66 + 3(surplus time)
    Number of Files        Implementation Time
    12                     2.59s  
    36                     3.79s
    108                    9.17s
    300                    25.73s
    900                    55.5s
    '''
    
    return 0.06*number_of_files**3 + 5.32*number_of_files**2 - 6.35*number_of_files + 1.66 + 2

# local 127.0.0.1
rb = rblib.Robot('127.0.0.1', 12345) #192.168.0.23

rb.open()
IOinit( rb )
rb.acq_permission()

# 하이퍼 파라미터 데이터 불러오기
hyper_file = open('./config/remote_config.json')
hyper_params = json.load(hyper_file)[sys.argv[2]][0] #sys.argv[2] = 'data_generation_HS'
hyper_file.close()


# 초기 위치, 속도, 가감속 시간 설정
pulseHome = [0,0,0,0,0,0]
time.sleep(0.5)
init_speed = float(hyper_params['speed_allow'])
init_acct = float(hyper_params['init_acc_time'])
init_dacct = init_acct
# rb.setvenv(3, 0.0, 0.0)
rb.plsmove(pulseHome[0], pulseHome[1], pulseHome[2], pulseHome[3], pulseHome[4], pulseHome[5], init_speed, init_acct, init_dacct)
time.sleep(0.5)

# HyperParameter Setting from json file
inertia_setting = list(map(int, hyper_params['inertia_gain'].split(',')   ))#0x5050
dumper_setting = list(map(int, hyper_params['damper_gain'].split(',')))  #0x5051
pole1_setting = list(map(int, hyper_params['first_pole'].split(',')))#0x5054
pole2_setting = list(map(int, hyper_params['second_pole'].split(','))) #0x5055
polei_setting = list(map(int, hyper_params['integral_pole'].split(','))) #0x5057

# Acceleration Setting
acc_setting = list(map(float, hyper_params['acc_time'].split(',')))
n_counts = len(acc_setting)

# Inertia Setting
inertia_num = int(sys.argv[1])  # sys.argv[1] = 0

# Reciprocating Motion with no acceleration
axis_num = int(sys.argv[3]) #sys.argv[3] = 1
degree = float(sys.argv[4]) #sys.argv[4] = 10
reduction_ratio = int(sys.argv[5]) #sys.argv[5] = 1
full_axis = 6 #sys.argv[6] = 6

pulse_value = 15360 /360 * degree * reduction_ratio

pulse = [pulse_value if i == axis_num-1 else 0 for i in range(full_axis)]

# Compress and Remove Data
# 필요한 용량 (바이트)

for inertia in inertia_setting:

    for dumper in dumper_setting:

        for pole1 in pole1_setting:

            for pole2 in pole2_setting:

                for poleI in polei_setting:
                    
                    print '-'*50
                    print "%d, %d, %d, %d, %d"%(int(inertia), int(dumper), int(pole1), int(pole2), int(poleI))
                    print ["ethercat","download", "-p", str(axis_num), "0x5050", "0", inertia]
                    subprocess.Popen(["ethercat","download", "-p", str(axis_num), "0x5050", "0", str(inertia)]) # change the parameter setting of slave
                    subprocess.Popen(["ethercat","download", "-p", str(axis_num), "0x5051", "0", str(dumper)])
                    subprocess.Popen(["ethercat","download", "-p", str(axis_num), "0x5054", "0", str(pole1)])
                    subprocess.Popen(["ethercat","download", "-p", str(axis_num), "0x5055", "0", str(pole2)])
                    subprocess.Popen(["ethercat","download", "-p", str(axis_num), "0x5057", "0", str(poleI)])
                    #subprocess.Popen(["ethercat","download",])
                    time.sleep(1)
                    #subprocess.Popen(["ethercat","upload", "-p", "3", "0x5050", "0"])
                    rb.set_parameter(0x1<<(axis_num-1)) #change the parameter setting of driver
                    now = datetime.datetime.now()
                    
                    for acc_time in acc_setting:
                        fname = "../generation_test/I.R_{0:04d}_D.R_{1:04d}_F.P_{2:04d}_S.P_{3:04d}_I.P_{4:04d}_A.T_{5:.3f}_R.I_{6:04d}_step_sample_{7}.csv".format(
                                                int(inertia), int(dumper), int(pole1), int(pole2), int(poleI),
                                                float(acc_time), int(inertia_num), now.strftime('%y%m%d_%H%M%S') )
                        subp = subprocess.Popen(["python","get55555sp_for_running.py", fname, "3000"])
                        time.sleep(1)
                        print "Measures!!!!!!!!!!!!!!!!!!!!"
         
                        res = rb.ioctrl(0, int(0x10000000), int(0x0000FFFF), int(0), int(0xFFFFFFFF))
                        # wordno, dataL, maskL, dataH, maskH  :  0, 0x00010000, 0xfffeffff, 0, 0xffffffff

                        rb.plsmove(pulse[0] ,pulse[1], pulse[2], pulse[3], pulse[4], pulse[5], 100, acc_time, acc_time)
                        time.sleep(0.3)
                        
                        rb.plsmove(pulseHome[0], pulseHome[1], pulseHome[2], pulseHome[3], pulseHome[4], pulseHome[5], 100, acc_time, acc_time)
                        time.sleep(0.3) 
                        res = rb.ioctrl(0, int(0x00000000), int(0x0000FFFF), int(0), int(0xFFFFFFFF))
                        time.sleep(0.1)
                        subp.terminate()
                        
                        print('_'*50)

    # 만약에 svstat가 servo-off를 의미하는 것이면, 해당 상태의 parameter를 출력...(첫번째 값을 보면 되도록)
    if not svstat:
        print('Current Parameter Setting : {0:04d}_{1:04d}_{2:04d}_{3:04d}_{4:04d}_a_{5:.3f}_i_{6:04d}'.format(
                                                int(inertia), int(dumper), int(pole1), int(pole2), int(poleI),
                                                float(acc_time), int(inertia_num)))
        print('Error Occurs!')
        sys.exit(1) 

    # File Download
    result = subprocess.Popen("df /dev/root | tail -1 | awk '{print $5}' | sed -e 's/%//'".encode(), stdout = subprocess.PIPE, shell = True)
    output, _ = result.communicate()
    # '21\n'
    # If output is over 60%, then pause this process
    print('Output')
    print('Percentage : {0}'.format(output))
    print('-----')
    if (int(output) > 40) or (inertia_setting.index(inertia) == len(inertia_setting)-1):
        print('Warning Percentage : {0}, need to transfer files'.format(output))
        print('Pause the process')
        # Waiting Time Function of the number of Files
        
        # location of data generation
        number_of_files = len(glob.glob('/home/generation_test/*.csv'))
        sleep_time = cal_sleep_time(number_of_files)

        # Reduce the noise of motor when it is not in use
        subprocess.Popen(["ethercat","download", "-p", str(axis_num), "0x5050", "0", str(150)]) 
        subprocess.Popen(["ethercat","download", "-p", str(axis_num), "0x5051", "0", str(100)])
        subprocess.Popen(["ethercat","download", "-p", str(axis_num), "0x5054", "0", str(25)])
        subprocess.Popen(["ethercat","download", "-p", str(axis_num), "0x5055", "0", str(200)])
        subprocess.Popen(["ethercat","download", "-p", str(axis_num), "0x5057", "0", str(200)])
        time.sleep(1)
        rb.set_parameter(0x1<<(axis_num-1))

        time.sleep(sleep_time)
        print('Resume the process')
        continue
    
    

                        
rb.close()