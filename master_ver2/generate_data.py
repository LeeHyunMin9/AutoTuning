import numpy as np
import sys
import json
import telnetlib
import subprocess
import ftplib
import os
import time
from file_transfer import testfile_upload


def make_pulse(params_dir, code_dir, trial_number, exp_condition, current_inertia, 
               host_num, telnet_port, axis_num, degree, reduction_ratio, full_axis):    
    '''
    입력 파라미터 
        - params_dir    : Controller 내의 파라미터 업로드 디렉토리
        - code_dir      : Controller 내의 데이터 생성을 제어 하는 코드를 업로드하는 디렉토리
        - trial_num     : 몇 번째 반복 실험인지
        - exp_condition : 어떤 실험(데이터 생성, 민감도 분석, 이상치 분석 등등)인지에 대한 명명
        - current_inertia : 현재 이너샤 세팅
        - host_num, telnet_port :  telnet 접속 관련 파라미터
        - axis_num, degree, reduction_ratio, full_axis : 실험 조건 관련 파라미터
    
    외부 함수
        - testfile_upload(dir) : Contoller 내의 업로드 할 파일(dir)의 폴더를 생성 및 업로드

    실행 결과 : 아래와 같은 구조로 데이터 생성
        - data
           - params_condition1
             - trial_num1
               - raw_data.csv
           - params_condition2
             - trial_num1
               - raw_data.csv
    '''
    
    # Upload Json File related to Parameter Setting & Data Generation File
    testfile_upload(params_dir)
    testfile_upload(code_dir)

    # Run Date Generation File and Transfer Generated Data to Local Directory
    telnet = telnetlib.Telnet(host_num, telnet_port, 5)    # (host = "192.168.0.23", port = 23, timeout = 5)
    telnet.read_until(b"login:", 5)
    telnet.write("{}\n".format("i611usr").encode())
    telnet.read_until(b"Password:", 5)
    telnet.write("{}\n".format("i611").encode())
    telnet.read_until(b"$", 5)
    telnet.write("python code_test.py {} {} {} {} {}\n".format(current_inertia, exp_condition, axis_num, degree, reduction_ratio).encode())
    while True:
        k = telnet.read_until(b"Output",3)
        if k != b'':
            # 파일 전송 프로세스 실행, 다만 터미널에서 엔터키를 누르지 않으면 실행이 안되는 문제가 있음
            
            if "Pause the process" in k.decode():
                print(k.decode())
                print('Start the file transfer process')
                current_time = time.time()
                subp = subprocess.Popen(["python", "./file_transfer.py", trial_number, axis_num], stdin = subprocess.PIPE, stdout=subprocess.PIPE)
                output, _ = subp.communicate()
                print(output.decode())
                print('Finish the file transfer process')
                print('Elapsed Time : {0:.4f}s'.format(time.time() - current_time))
                subp.terminate()
                
        else:
            print('End')
            break
    
    # reset.py 실행
    # command 가 python code_test.py 를 포함하고 있는 process kill
        
    #telnet.write("ps -ef | grep 'python code_test.py' | awk '{print $2}' | xargs kill -9\n".encode())
    telnet.close()


if __name__ == '__main__':
    
    '''
    파라미터 실험 셋팅을 서버에 전송하여 PLS data를 생성한 다음 로컬의 지정된 directory에
     raw_data.csv 을 전송한다. 이후에 데이터 전처리 과정을 수행한 다음,
    raw_data.png, properties.txt, classifying.txt  

    파라미터 실험 셋팅
        - (파일명) 날짜.json
        - (데이터) dict(key = params_name, value = params_values)

    데이터 생성 포맷
        - (파일명) paramas_values_날짜.csv
        - (데이터) PLS data(cmd pulse/rpm/torque, fdb pulse/rpm/torque, sensor value 등)

    지정된 로컬 디렉토리
        - (파일명) params_values/trial_0/
        - (저장 작업) 위의 PLS data를 파라미터 값의 배치에 해당하는 디렉토리 및에 raw_data.csv로 저장
    
    전처리 과정 실행 및 데이터 관련 정보 추가 저장
        - (raw_data.png)   raw_data.csv에서 유의미한 특성값을 가져온 후, 필요없는 시간을 자른다음
                            특성값이 포함되도록 그래프를 그린 이후 저장한다.
        - (properties.json) raw_data 의 특성값을 저장한다.
        - (classifying.json) raw_data 가 이상치의 특성을 가지고 있는지에 대한 라벨링 결과를 저장한다.
    '''

    # 로봇 통신 관련 파라미터
    host_num = '192.168.0.23'
    #host_port = 55555
    telnet_port = 23
    # 업로드 파일 디렉토리
    params_dir = f'./config/{sys.argv[3]}'
    code_dir = './remote_server/code_test.py'
    # 실험 시행 번호 관련 파라미터
    trial_number = sys.argv[1]
    current_inertia = sys.argv[2]
    exp_condition = sys.argv[4]

    axis_num = sys.argv[5]
    degree = sys.argv[6]
    reduction_ratio = sys.argv[7]
    full_axis = sys.argv[8]
    
    # 실험 조건이 정해지지 않은 경우 data_generation 을 수행
    with open(params_dir, "r") as file:
        key = json.load(file).keys()

    if exp_condition not in key:
        exp_condition = 'data_generation_sp1'    


    # 데이터 생성 프로세스
    make_pulse(params_dir, code_dir, trial_number, exp_condition, current_inertia, host_num, telnet_port,
               axis_num, degree, reduction_ratio, full_axis)

