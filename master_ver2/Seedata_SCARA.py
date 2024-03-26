#! /usr/bin/python
# -*- coding: utf-8 -*-

import socket
import sys
import os
import matplotlib.pyplot as plt
import easygui
import numpy as np
import csv
import tkinter
import re
axis_num = 2

# axis = [ 0, 1, 2, 3]

axis = [1]
#axis = [i for i in range(axis_num)]
# axis = [ 0]
# axis = [ 0, 1]
# axis = [ 2, 3]
# mul = [ 1, 1, 1, 1]
mul = [ 1 for i in range(axis_num)]
# mul = [ 2, 1, 1, 1]
# mul = [ 1, 1, 50, 100]

colorMap = ["peru","orangered","orange","yellow","limegreen","deepskyblue","violet","gray","white"] #, "cyan","magenta"]
# legendMap = ["j1", "j2", "j3", "j4", "j5", "j6"]
legendMap = ["j{}".format(i+1) for i in range(axis_num)]

legendMap1 = ["j1_1", "j2_1", "j3_1", "j4_1", "j5_1", "j6_1"]
legendMap2 = ["j1_2", "j2_2", "j3_2", "j4_2", "j5_2", "j6_2"]

colorM = ['r','b','g','c','m','k']

plt.rcParams["figure.figsize"] = (16,12)
plt.rcParams['axes.grid'] = True 

array_integer_Adr = {                       # When 6 Axis
    "time_stamp_Adr":0,                     #      "time_stamp_Adr":    0,
    "mani   p_power_Adr":1,                    #     "manip_power_Adr":    1,
    "svsts_Adr":2,                          #           "svsts_Adr":    2,
    "verbsts_Adr":3,                        #         "verbsts_Adr":    3,
    "cmdpls_Adr":7+axis_num*0,              #          "cmdpls_Adr":    7,  (7 + 4*0 )
    "fbpls_Adr":7+axis_num*1,               #           "fbpls_Adr":   11,  (7 + 4*1 )
    "force_sensor_Adr":7+axis_num*2,        #    "force_sensor_Adr":   15,  (7 + 4*2 )  <-- 6
    "torque_Adr":7+axis_num*2+6,            #          "torque_Adr":   21,  (7 + 4*2 + 6)
    "filter_torque_Adr": 7+axis_num*3+6,    #   "filter_torque_Adr":   25,  (7 + 4*3 + 6)
    "xyz_Adr":7+axis_num*4+6 ,              #             "xyz_Adr":   29,  (7 + 4*4 + 6)  <-- 6
    "accl_Adr":7+axis_num*4+6*2,            #            "accl_Adr":   35,  (7 + 4*4 + 6*2 )  <-- 3
    "tempAdr":7+axis_num*4+6*2+3 ,          #             "tempAdr":   38,  (7 + 4*4 + 6*2 + 3) # 0x523e
    "thermal_torque_Adr":7+axis_num*5+6*2+3,#  "thermal_torque_Adr":   42,  (7 + 4*5 + 6*2 + 3) # 0x523d
    "volage_Adr":7+axis_num*6+6*2+3 ,       #          "volage_Adr":   46,  (7 + 4*6 + 6*2 + 3) # 0x523f
    "board_tempAdr":7+axis_num*7+6*2+3 ,    #       "board_tempAdr":   50,  (7 + 4*7 + 6*2 + 3) # 0x523e
    "target_pos_Adr":7+axis_num*8+6*2+3 ,   #      "target_pos_Adr":   54,  (7 + 4*8 + 6*2 + 3) # 0x5232
    "402_actpos_Adr":7+axis_num*9+6*2+3 ,   #      "402_actpos_Adr":   58,  (7 + 4*8 + 6*2 + 3) # 0x6064
    "402_flwerr_Adr":7+axis_num*10+6*2+3 ,  #      "402_flwerr_Adr":   62,  (7 + 4*8 + 6*2 + 3) # 0x60f4
}

array_string_Adr = {                       
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

last_integer_Adr_index = axis_num
bit_array_Adr = {
    "system_io"         : 0,
    "holdinfo_flags"    : 1,
    "_402ctrl"          : 2,
    "_402err"           : 2 + axis_num*1,
    "_sts"              : 2 + axis_num*2,
    "_stdio"            : 2 + axis_num*3,
    "_armio"            : 2 + axis_num*3 + 1
}
last_bit_Adr_index = 1

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
  
    def __init__(self, dataAdr, colorAdr, axis_num):
        self.colorMap = colorAdr
        self.arrayMap = dataAdr
        self.axisnum = axis_num
        self.N_int_datas = max(array_integer_Adr.values()) + last_integer_Adr_index
        self.N_datas = max(array_integer_Adr.values()) + last_integer_Adr_index + max(bit_array_Adr.values()) + last_bit_Adr_index

    def puttingData(self, fileNames):
        self.fileName = fileNames
        with open(fileNames, 'r' ) as f:
            reader = csv.reader(f)                     # 한번 실행할때마다 한 줄씩 읽어 옴.
            self.header = next(reader)                 # 첫줄은 header로 함.
            print (reader)
            # make empty array
            self.int_datas  = np.zeros((5000000,self.N_int_datas))            # N_int_datas(60)개의 데이터를 500만개 저장 할 수 있는 저장 공간을 생성. # axis_num 6개 기준
            self.bit_datas  = []
            j = 0
            # read data from csv file.
            for self.row in reader:                    # data 한줄씩 읽어 row에 넣기
                self.buf_int_datas  = []
                self.buf_bit_datas = []
                for data_index, read_data in enumerate(self.row):  # 한줄씩 읽은 데이터를 r에 다가 저장해 아래의 명령들을 실행  
                    if not( self.N_datas < data_index ):                  # i가 N_datas를 넘지 않으면...(N_datas(85)는 한줄에 저장된 데이터의 개수.) # axis_num 6개 기준
                        if self.checkFloat(read_data) == True:
                            self.buf_int_datas.append(float(read_data)) # 값이 float이면 buf_int_datas 저장
                        else:
                            self.buf_bit_datas.append(read_data)       # 값이 float이 아니면 buf_bit_datas 저장
                self.notEnoughNumber = self.N_int_datas - len(self.buf_int_datas)  # buf_int_datas는 현재 self.row의 float값 만을 저장하고 있음 
                for i in range(self.notEnoughNumber):
                    self.buf_int_datas.append(self.buf_int_datas[0])            # 60을 맞추기 위한 데이터 추가
                self.int_datas[j, :self.N_int_datas] = self.buf_int_datas[ :self.N_int_datas]     # int_datas는 float값만 이 있는 buf_int_datas값을 저장
                self.bit_datas.append(self.buf_bit_datas)               # bit_datas에는 float값이 아닌 것을 모은 buf_int_datas를 저장
                j = j+1                                         # 다음 값을 줄로 감
            # place each parameter. based on int_datas, bit datas.
            self.int_datas = self.int_datas[0:j-1]      # N_int_datas(60)개를 저장하는 500만개의 데이터에서 현재 저장된 데이터 개수만큼만의 데이터로 줄임. # axis_num 6개 기준
            self.dataCollectionF(self.int_datas, self.arrayMap) # putting data in each axis
        return self.int_datas, self.bit_datas
    
    def checkFloat(self,read_data):
        try:
            float(read_data)
        except:
            return False
        return  True
    
    def dataCollectionF(self, integer_datas, parameterAdress):
        axis_param  = self.axisnum
        xyz_param   = 6
        accel_param = 3
        float_datas = integer_datas.T       # transpose the array : row <-> col
        Nrow = np.shape(float_datas)[1]

        # if data is only one axis then putting right away
        self.time_stamp         = float_datas[ parameterAdress["time_stamp_Adr"], :]
        #self.manip_power        = float_datas[ parameterAdress["manip_power_Adr"], :]
        self.svsts              = float_datas[ parameterAdress["svsts_Adr"], :]
        self.verbsts            = float_datas[ parameterAdress["verbsts_Adr"], :]

        # make empty array(not one address case)
        self.cmdpls             = np.zeros([ axis_param, Nrow], dtype=float)
        self.fbpls              = np.zeros([ axis_param, Nrow], dtype=float)
        self.errpls             = np.zeros([ axis_param, Nrow], dtype=float)
        self.force_sensor       = np.zeros([  xyz_param, Nrow], dtype=float)
        self.torque             = np.zeros([ axis_param, Nrow], dtype=float)
        self.filter_torque      = np.zeros([ axis_param, Nrow], dtype=float)
        self.xyz                = np.zeros([  xyz_param, Nrow], dtype=float)
        self.acceleration       = np.zeros([accel_param, Nrow], dtype=float)
        self.temperature        = np.zeros([ axis_param, Nrow], dtype=float)
        self.thermal_torque     = np.zeros([ axis_param, Nrow], dtype=float)
        self.volage             = np.zeros([ axis_param, Nrow], dtype=float)
        self.board_temp         = np.zeros([ axis_param, Nrow], dtype=float)
        self.target_pos         = np.zeros([ axis_param, Nrow], dtype=float)
        self.actpos_402         = np.zeros([ axis_param, Nrow], dtype=float)
        self.flwerr_402         = np.zeros([ axis_param, Nrow], dtype=float)
        # self.fbxyz              = np.zeros([  xyz_param, Nrow], dtype=float)

        # putting data(for each axis)
        for axis_index in range(axis_param):
            self.cmdpls[axis_index]         = float_datas[ parameterAdress["cmdpls_Adr"]        + axis_index]
            self.fbpls[axis_index]          = float_datas[ parameterAdress["fbpls_Adr" ]        + axis_index]
            self.errpls[axis_index]         = float_datas[ parameterAdress["cmdpls_Adr"]        + axis_index] - float_datas[ parameterAdress["fbpls_Adr" ] + axis_index] 
            self.torque[axis_index]         = float_datas[ parameterAdress["torque_Adr"]        + axis_index]
            self.filter_torque[axis_index]  = float_datas[ parameterAdress["filter_torque_Adr"] + axis_index]
            self.xyz[axis_index]            = float_datas[ parameterAdress["xyz_Adr"]           + axis_index]
            self.temperature[axis_index]    = float_datas[ parameterAdress["tempAdr"]           + axis_index]
            self.thermal_torque[axis_index] = float_datas[ parameterAdress["thermal_torque_Adr"]+ axis_index]
            self.volage[axis_index]         = float_datas[ parameterAdress["volage_Adr"]        + axis_index]
            self.board_temp[axis_index]     = float_datas[ parameterAdress["board_tempAdr"]     + axis_index]
            self.target_pos[axis_index]     = float_datas[ parameterAdress["target_pos_Adr"]    + axis_index]
            self.actpos_402[axis_index]     = float_datas[ parameterAdress["402_actpos_Adr"]    + axis_index]
            self.flwerr_402[axis_index]     = float_datas[ parameterAdress["402_flwerr_Adr"]    + axis_index]

        for xyz_index in range(xyz_param):
            self.force_sensor[axis_index]   = float_datas[ parameterAdress["force_sensor_Adr"]          + xyz_index]
            self.xyz[axis_index]                = float_datas[ parameterAdress["xyz_Adr"]               + xyz_index]
            # self.fbxyz[axis_index]            = float_datas[ parameterAdress["fbxyz_Adr"]             + axis_index]
        for accel_index in range(accel_param):
            self.acceleration[accel_index]  = float_datas[ parameterAdress["accl_Adr"]    + accel_index]

def bitting(IOdatas):
    bit_result=[]
    for hexascii in IOdatas:
        if hexascii != "":
            bit_list  = [int(i) for i in bin(int(hexascii, 16))[2:]] # hex 형태의 string을 int 형태로 변환하고 bin 형태로 변환후 리스트 형태로 변환
            reverse_bit = bit_list[::-1]                             # 숫자를 거꾸로 하기
            reverse_32bit = reverse_bit + [0]*(32-len(reverse_bit))  # 남는 자리를 0으로 채워 32자리수를 채우기
            bit_result.append(list(reverse_32bit))                   # 밑으로 데이터 쌓기
    bit_output = np.array(bit_result).T                              # Transpose 하여 행과 열을 변환하기 앞쪽을 bit와 연결하기 위함.
    return bit_output

# select the file
root = tkinter.Tk()
root.filename =  tkinter.filedialog.askopenfilename(initialdir = "D:/001.Developement of Company Work/figure_ppt/MotorTest",title = "choose your file",filetypes = (("csv files","*.csv"),("all files","*.*")))

read_address = root.filename
root.destroy()

data = DataCollector(array_integer_Adr, colorMap, axis_num)
integer_datas,bit_datas = data.puttingData(read_address)

read_address = read_address.replace('\\', '/')
print(read_address)
#cwd = os.getcwd().replace('\\', '/')
#cwd = "D:/001.Developement of Company Work/data/AutoTuning"
cwd = "D:/001.Developement of Company Work/figure_ppt/MotorTest"
mat = re.match(cwd, read_address)
select_file = read_address[mat.end()+1:]
## integer_datas ##
# xxx.(option)
# ---option------
# time_stamp         
# manip_power        
# svsts              
# verbsts            
# cmdpls             
# fbpls              
# errpls             
# torque             
# filter_torque      
# xyz                
# temperature        
# fbxyz              
# acceleration      

time_stamp_zero_start           = (data.time_stamp - data.time_stamp[0])/1000/1000
time_stamp_zero_start_vel       = (data.time_stamp[:-1] - data.time_stamp[0])/1000/1000

cmd_rpm                         = np.diff(data.cmdpls)*1000*60/15360
fb_rpm                          = np.diff(data.fbpls)*1000*60/15360

############################# first plotting ###########################
plt.cla()
fig = plt.figure(1,clear = True)
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

ax1.set_title("pulse(1:cmd,2:fb,3:err)")
for axis_num in axis:
    ax1.plot(time_stamp_zero_start,data.cmdpls[axis_num]*mul[axis_num], label='cmd_{}'.format(axis_num))   # cmd
    ax1.plot(time_stamp_zero_start,data.fbpls[axis_num]*mul[axis_num], label='fb_{}'.format(axis_num))    # fb
    ax1.plot(time_stamp_zero_start,data.errpls[axis_num], label='err_{}'.format(axis_num))                   # err
ax1.legend()

ax2.set_title("rpm(1:cmd,2:fb)")
for axis_num in axis:
    ax2.plot(time_stamp_zero_start_vel,cmd_rpm[axis_num]*mul[axis_num], label='cmdrpm_{}'.format(axis_num))   # rpm(based on cmd)
    ax2.plot(time_stamp_zero_start_vel,fb_rpm[axis_num]*mul[axis_num], label='fbrpm_{}'.format(axis_num))   # rpm(based on cmd)
ax2.legend()

ax3.set_title("torque(1:torque,2:filter_torque)")
for axis_num in axis:
    ax3.plot(time_stamp_zero_start,data.torque[axis_num]*mul[axis_num], label='torq_{}'.format(axis_num))   # cmd
    ax3.plot(time_stamp_zero_start,data.filter_torque[axis_num]*mul[axis_num], label='filtorq_{}'.format(axis_num))   # cmd
ax3.legend()
plt.show()

while True:
    start_inp = input("start_time?")
    end_inp = input("end_time?")

    if start_inp != '':
        start_time = float(start_inp)
        for i in range(len(data.time_stamp)):
            if(time_stamp_zero_start[i] > start_time):
                start_index = i
                break
    else:
        start_index = 0

    if end_inp != '':
        end_time = float(end_inp)
        for i in range(len(data.time_stamp)):
            if(time_stamp_zero_start[i] > end_time):
                end_index = i
                break
    else:
        end_index = -1

    ############################# next plotting ###########################
    # plt.cla()
    fig = plt.figure(num = select_file, clear = True)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ax1.set_title("pulse(1:cmd,2:fb,3:err)")
    for axis_num in axis:
        ax1.plot(time_stamp_zero_start[start_index:end_index],data.cmdpls[axis_num][start_index:end_index]*mul[axis_num], label='cmd_{}'.format(axis_num))   # cmd
        ax1.plot(time_stamp_zero_start[start_index:end_index],data.fbpls[axis_num][start_index:end_index]*mul[axis_num], label='fb_{}'.format(axis_num))    # fb
        ax1.plot(time_stamp_zero_start[start_index:end_index],data.errpls[axis_num][start_index:end_index]*mul[axis_num], label = 'err_{}'.format(axis_num))  # err
    ax1.legend()

    ax2.set_title("rpm(1:cmd,2:fb)")
    for axis_num in axis:
        ax2.plot(time_stamp_zero_start_vel[start_index:end_index],cmd_rpm[axis_num][start_index:end_index]*mul[axis_num], label='cmdrpm_{}'.format(axis_num))   # rpm(based on cmd)
        ax2.plot(time_stamp_zero_start_vel[start_index:end_index],fb_rpm[axis_num][start_index:end_index]*mul[axis_num], label='fbrpm_{}'.format(axis_num))   # rpm(based on fb)
    ax2.legend()

    ax3.set_title("torque(1:torque,2:filter_torque)")
    for axis_num in axis:
        ax3.plot(time_stamp_zero_start[start_index:end_index],data.torque[axis_num][start_index:end_index]*mul[axis_num], label='torq_{}'.format(axis_num))   # cmd
        ax3.plot(time_stamp_zero_start[start_index:end_index],data.filter_torque[axis_num][start_index:end_index]*mul[axis_num], label='filtorq_{}'.format(axis_num))   # fb
    ax3.legend()

    # max_value = 0
    # ax2.set_title("torque(1:torque_temp,2:filter_torque, 3.voltage)")
    # for axis_num in axis:
    #     ax2.plot(time_stamp_zero_start[start_index:end_index],data.torque_temp[axis_num][start_index:end_index]*mul[axis_num], label='torqtemp_{}'.format(axis_num))   # cmd
    #     ax2.plot(time_stamp_zero_start[start_index:end_index],data.filter_torque[axis_num][start_index:end_index]*mul[axis_num], label='filtorq_{}'.format(axis_num))   # cmd
    #     # if( max_value < max(data.filter_torque[axis_num][start_index:end_index]*mul[axis_num])):
    #     #     max_value = max(data.filter_torque[axis_num][start_index:end_index]*mul[axis_num])
    #     # ax2.plot(time_stamp_zero_start[start_index:end_index],data.voltage[axis_num][start_index:end_index]*mul[axis_num], label='voltage_{}'.format(axis_num))   # voltage
    
    # # ax2.plot(time_stamp_zero_start[start_index:end_index],data.ref_voltage[0][start_index:end_index]*mul[axis_num], label='40V')   # voltage
    # # ratio = max_value / max(data.svsts[start_index:end_index])
    # # ax2.plot(time_stamp_zero_start[start_index:end_index],data.svsts[start_index:end_index]*ratio, label='svsts')   # cmd

    ax2.legend()

    plt.show()
