#! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright(C) 2016-2017 NIDEC CORPORATION All rights reserved.

import socket
import sys
import os
import signal

#---------------
import rblib
# Neativeとの通信用
_RBLIB_HOST = "127.0.0.1"
_RBLIB_PORT = 12345
#---------------

def_OUTFILE_NAME = "data55555.csv"
def_GET_TIME = 1*1000 #ms
get_data_flag = True

def handler(signum, frame):
    global get_data_flag
    print("Ctrl+C")
    get_data_flag = False

signal.signal(signal.SIGTERM, handler)


def main():
    global get_data_flag
    get_data_flag = True
    #host = "192.168.0.23"
    host = "127.0.0.1"
    port = 55555
    bufsize = 4096

    #---------------
    # Nativeとの通信用
    rbl = rblib.Robot(_RBLIB_HOST, _RBLIB_PORT)
    rbl.open()
    #---------------

    if len(sys.argv)>1:
        OUTFILE_NAME = sys.argv[1]
        GET_TIME = int(sys.argv[2])
    else:
        OUTFILE_NAME = def_OUTFILE_NAME
        GET_TIME = def_GET_TIME

    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    size = 0
    cnt = 0
    data = ""
    try:
        sock.connect((host, port))
        while cnt < GET_TIME and get_data_flag: # ms
            recv_data = sock.recv(bufsize)
            cnt += recv_data.count("\n")
            data += recv_data
    finally:
        f = open(OUTFILE_NAME,"w")
        f.write(data)
        f.close()
        #---------------
        # Nativeに55555ポートのClose処理を指令する
        rbl.sysctrl(3, 0)
        rbl.close()
        #---------------
        sock.close()
        
if __name__ == "__main__":
    main()
#eof
