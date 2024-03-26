#!/usr/bin/python
#-*- coding: utf-8 -*-

import time
import subprocess
import datetime

import rblib

def svstat(rb):
    res = rb.ioctrl(128, 0, 0xffffffff, 0, 0xffffffff)
    if res[0] == False:
        print("Unable to read servo power")
        raise Exception
    if (res[1] & 0x01)!=0:
        return 1
    return 0


rb = rblib.Robot('127.0.0.1', 12345) #127.0.0.1

rb.open()
rb.acq_permission()

pulseHome = [0,0,0,0,0,0]
time.sleep(0.5)
init_speed = 5
init_acct = 1.0
init_dacct = 1.0
# rb.setvenv(3, 0.0, 0.0)
rb.plsmove(pulseHome[0], pulseHome[1], pulseHome[2], pulseHome[3], pulseHome[4], pulseHome[5], init_speed, init_acct, init_dacct)
time.sleep(0.5)

# rb.svctrl(2)
# time.sleep(2)

# サーボOFFになるまで待つ
# if svstat(rb) == 1:
#     print("Waiting for servo off...")
#     while svstat(rb) == 1:
#         time.sleep(0.5)

# rb.relbrk(4)
# y = raw_input("check parameter.")
# rb.clpbrk(4)

# # サーボON待ち
# print ("Please turn servo power on to continue")
# while svstat(rb) != 1:
#     time.sleep(0.5)

interia_setting = [100]#[200, 250, 300, 350,  450, 600, 750, 900]
second_pole_setting = [200, 300, 400, 500, 600]
# Reciprocating Motion with no acceleration
# pulse = [0, 15360/360*10, 0, 0, 0, 0]  #Ratio of Deacceleration not 101(15360*101/360)

axis_num = 2
gear_ratio = 1
degree = 10
pulse_value = 15360/360*degree*gear_ratio
full_axis = 6
speed_ratio = 5
acc_time = 1.0
pulse = [ pulse_value if i == axis_num-1 else 0 for i in range(full_axis) ]
print(pulse)

for intertia in interia_setting:
# while True:
    # intertia = raw_input("servo gain ?")
    print "%d"%(int(intertia))
    print ["ethercat","download", "-p", str(axis_num), "0x5050", "0", intertia]
    subprocess.Popen(["ethercat","download", "-p", str(axis_num), "0x5050", "0", str(intertia)])
    time.sleep(1)
    rb.set_parameter(0x1<<(axis_num-1))
    now = datetime.datetime.now()
    fname = "../generation_test/I.R_{0:04d}_S.P_{1:04d}_degree_{2:04d}_S.R_{3:04d}_A.T_{4:.3f}_{5:%y%m%d%H%M%S}_robot.csv".format(int(intertia), int(400), degree, speed_ratio, acc_time, now)
    # fpath = os.path.join(working_path, fname)
    subprocess.Popen(["python","get55555sp_for_running.py", fname, "2000"])
    time.sleep(0.5)
    print "Measure"

    for t in range(3):        
        rb.plsmove(pulse[0] ,pulse[1], pulse[2], pulse[3], pulse[4], pulse[5], speed_ratio, acc_time, acc_time)
        time.sleep(0.1)
        
        rb.plsmove(-pulse[0], -pulse[1], -pulse[2], -pulse[3], -pulse[4], -pulse[5], speed_ratio, acc_time, acc_time)
        #rb.plsmove(pulseHome[0], pulseHome[1], pulseHome[2], pulseHome[3], pulseHome[4], pulseHome[5], 100, 0.5, 0.5)
        time.sleep(0.1)

print "Restore Original Parameters"
rb.set_parameter(0x1<<(axis_num-1))
subprocess.Popen(["ethercat","download", "-p", str(axis_num), "0x5050", "0", "200"])
