#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import time
import ftplib
import sys
from utils.utils import logging_time


ftp_host = "192.168.0.23"
ftp_user = "i611usr"
ftp_password = "i611"


        

def testfile_upload(local_directory):
    # code_test.py, 파라미터 세팅 관련 파일 업로드
    # FTP 연결 및 파일 전송
    try:
        # ftp 연결 - context manager가 없음(python 2.7 이하로는)
        with ftplib.FTP() as ftp:
            ftp.connect(host=ftp_host,port=21)
            ftp.encoding = 'utf-8'
            s = ftp.login(user=ftp_user,passwd=ftp_password)
        
            ftp.cwd('/home/i611usr')  # 현재 폴더 이동

            def exist(fname):
                if fname in ftp.nlst():
                    return True
                return False

            def chdir(file_path):
                #for path in file_path.split('/'):
                # print(path)
                # if len(path) == 0:
                #     continue
                if exist(file_path):
                    ftp.cwd('/home/i611usr')
                else:
                 
                    ftp.mkd(file_path)
                    ftp.cwd('/home/i611usr')

            # 디렉토리 파일 생성 및 업로드
            if 'config' in local_directory:
                chdir('config')
                with open(file = local_directory, mode = 'rb') as wf:
                    ftp.storbinary('STOR {}'.format(local_directory), wf)

            if 'code_test' in local_directory:
                with open(file = local_directory, mode = 'rb') as wf:
                    ftp.storbinary('STOR {}'.format('./code_test.py'), wf)    
            # /remote_server/code_test.py -> code_test.py
            
    except Exception as e:
        print(e)


@logging_time
def data_download(trial_number, axis_num):
    local_directory = 'D:/001.Developement of Company Work/data/AutoTuning'
    motor_type = 'Small_Joint' if int(axis_num) == 1 else 'Zero_460W' # ['Hyeon_Seo','Small_Joint', 'Zero_460W']
    print("Start Downloading Process!")
    # FTP 연결 및 파일 전송
    try:
        # ftp 연결 - context manager가 없음(python 2.7 이하로는)
        with ftplib.FTP() as ftp:
            ftp.connect(host=ftp_host,port=21)
            ftp.encoding = 'utf-8'
            s = ftp.login(user=ftp_user,passwd=ftp_password)
        
            ftp.cwd('/home/generation_test')  # 현재 폴더 이동
        
            # 파일다운로드
            list = ftp.nlst()
            print(list)
            
            for file in list :
                # 저장 위치 : /data/parameter_setting/trial_{number}/raw_data.csv
                params_configuration_name = '_'.join(file.split('\\')[-1].split('_')[:14])
                # 저장 디렉토리 없으면 생성
                save_dir = os.path.join(local_directory, params_configuration_name, motor_type, f'trial_{trial_number}')
                os.makedirs(save_dir , exist_ok= True)

                # 저장 위치에 파일 생성
                with open(file=os.path.join(save_dir, 'raw_data.csv'), mode='wb') as rf:
                    ftp.retrbinary('RETR {}'.format(file), rf.write)

                # 서버(컨트롤러) 파일 삭제
                ftp.delete(file)

            # print("Removing Process!")
            # for file in list :
            #     if list:
            #         ftp.delete(file)

    except Exception as e:
        print(e)

if __name__ == '__main__':
    # sys.argv 있으면 받아오고, 없으면 내가 주는 값으로 테스트
    
    if len(sys.argv) > 1:
        trial_number = sys.argv[1]
        axis_num = sys.argv[2]
    else:
        trial_number = '0'
        axis_num = '2'
    
    data_download(trial_number = trial_number, axis_num = axis_num)