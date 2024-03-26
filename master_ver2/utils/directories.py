import os
import shutil
import glob
import numpy as np

def find_directory(root_directory, target_directory_name):
    '''
    target_directory_name : ["Hyeon_Seo", "Small_Joint"]
    root_directory
        - child_directory
           - child_directory
              - SJ
           - child_directory
              - SJ
    0000으로 끝나는 디렉토리를 찾아서 그 자식 폴더가 Small_Joint를 포함하지만 Hyeon_Seo를 포함하지 않는 디렉토리를 찾는다.
    '''
    target_directory_list = []    
    for directory in glob.glob(f'{root_directory}/*_0000'):
        if not os.path.isdir(os.path.join(directory, target_directory_name[0])) and os.path.isdir(os.path.join(directory, target_directory_name[1])):
            target_directory_list.append(directory.split('\\')[-1])
        
    print(target_directory_list)
    
def remove_directory(root_directory, old_name_list):
    '''
    root_directory
        - child_directory
           - child_directory
              - SJ
           - child_directory
              - SJ

    Remove SJ directories and their subfolders
    '''
    
    for dirpath, dirnames, files in os.walk(root_directory):
        # 디렉토리 삭제
        for index in range(len(old_name_list)):
            if old_name_list[index] in dirnames:
                del_dir_path = os.path.join(dirpath, old_name_list[index])
                shutil.rmtree(del_dir_path)
                print(f"디렉토리 및 하위 폴더를 삭제했습니다: {del_dir_path}")


def rename_directory(root_directory, old_name_list, new_name_list):
    for dirpath, dirnames, _ in os.walk(root_directory):
        # 디렉토리 이름 변경

        for index in range(len(old_name_list)):
            if old_name_list[index] in dirnames:
                old_dir_index = dirnames.index(old_name_list[index])
                dirnames[old_dir_index] = new_name_list[index]
                old_dir_path = os.path.join(dirpath, old_name_list[index])
                new_dir_path = os.path.join(dirpath, new_name_list[index])
                os.rename(old_dir_path, new_dir_path)
                print(f"디렉토리 이름을 변경했습니다: {old_dir_path} -> {new_dir_path}")

def rename_directory_2(data_list, old_name_list, new_name_list):
    '''
    (Before)
    - I.R_0150_D.R_0100_F.P_0010_S.P_0700_I.P_0400_A.T_0.070_R.I_0009
        - Small_Joint
    (After)
    - I.R_0150_D.R_0100_F.P_0010_S.P_0700_I.P_0400_A.T_0.070_R.I_0009
        - Zero_460W
    '''
    for directory in data_list:
        for index in range(len(old_name_list)):
            if os.path.isdir(directory):
                if old_name_list[index] in os.listdir(directory):
                    old_dir_path = os.path.join(directory, old_name_list[index])
                    new_dir_path = os.path.join(directory, new_name_list[index])
                    os.rename(old_dir_path, new_dir_path)
                    print(f"디렉토리 이름을 변경했습니다: {old_dir_path} -> {new_dir_path}")

def rename_directory_3(data_list):
    '''
    (Before)
    - I.R_0150_D.R_0100_F.P_0010_S.P_0700_I.P_0400_A.T_0.010_R.I_0000
    (After)
    - I.R_0150_D.R_0100_F.P_0010_S.P_0700_I.P_0400_A.T_0.010_R.I_0009
    if target directory already exists, then move the directory to the target directory
    else, just rename the directory name to target directory name
    '''
    for directory in data_list:
        if os.path.isdir(directory):
            new_dir_path = directory.replace('R.I_0000', 'R.I_0009')
            if os.path.exists(new_dir_path):
                # 디렉토리가 이미 존재할 때, 해당 내용들을 이동시킨다.
                files = os.listdir(directory)
                for file in files:
                    shutil.move(os.path.join(directory, file), os.path.join(new_dir_path, file))
                os.rmdir(directory)
            else:
                os.rename(directory, new_dir_path)
                print(f"디렉토리 이름을 변경했습니다: {directory} -> {new_dir_path}")
    

def move_directory(root_directory, target_directory_name):
    '''
    AutoTuning
        - I.R_0150_D.R_0100_F.P_0010_S.P_0700_I.P_0400_A.T_0.070_R.I_0003
            - trial_0
        
    1. Make a new directory(Hyeon_Seo) with upper structure and move the trial_0 directory to the new directory    
    2. Merge Hyeon_Seo directories and trial_0 directories and set like this form
       (Before) -I.R_0150_D.R_0100_F.P_0010_S.P_0700_I.P_0400_A.T_0.070_R.I_0003
                    - Hyeon_Seo
                        - raw_data.csv
                    - trial_0
       (After)  -I.R_0150_D.R_0100_F.P_0010_S.P_0700_I.P_0400_A.T_0.070_R.I_0003
                    - Hyeon_Seo
                        - trial_0
                            - raw_data.csv
    '''
    # 예외 directory 생성

    write_mode = False
    delete_mode = False
    if not os.path.exists(f'{root_directory}/exception.txt'):
        log_file = open(f'{root_directory}/exception.txt', 'wb')
    
    total_directory= glob.glob(f'{root_directory}/*')
    error_idx = []   
    
    if os.path.exists(f'{root_directory}/exception.txt'):
        read_file = open(f'{root_directory}/exception.txt', 'rb')
        lines = read_file.readlines()
        for index, err_msg in zip(lines[::2], lines[1::2]):
            
            index = index.decode('utf-8').rstrip('\r\n')
            
            if index != '' or index != '\n' or index != None:
                error_idx.append(total_directory.index(index.replace('\\\\', '\\')))
        
        read_file.close()

    if len(error_idx) > 0:
        total_directory = list(np.array(total_directory)[error_idx])
    
    for directory in total_directory:
        
        if not os.path.isdir(directory):
            continue
        
        try:
            # Merge Hyeon_Seo directories and trial_0 directories like above form
            new_directory_path = os.path.join(directory, target_directory_name)
                  
            if len(os.listdir(directory)) == 2:
                # ['Hyeon_Seo', 'trial_0'], ['Hyeon_Seo', 'trial_1'] 형태(trial 내부의 파일이 옮겨지지 않은 상황일때
                
                
                trial_dirs = os.listdir(directory)
                trial_dirs.remove(target_directory_name)
                trial_dir_name = trial_dirs[0]
                empty_directory_path = os.path.join(directory,trial_dir_name)
                files = os.listdir(empty_directory_path)
                #shutil.move(empty_directory_path, new_directory_path)
                os.makedirs(os.path.join(new_directory_path, target_directory_name, trial_dir_name), exist_ok= True) # directory/Hyeon_Seo/trial_0
                
                for file in files:
                    shutil.move(os.path.join(new_directory_path, file), os.path.join(new_directory_path, trial_dir_name, file))
                os.rmdir(empty_directory_path) # directory/trial_0 삭제
                print(f"미완된 디렉토리를 전환했습니다: {empty_directory_path} -> {os.path.join(new_directory_path, trial_dir_name)}")


            if not os.path.exists(new_directory_path):
                # Hyeon_Seo 디렉토리가 존재하지 않을 때
                trial_dirs = os.listdir(directory)
                os.makedirs(new_directory_path, exist_ok=True)              # directory/Hyeon_Seo
                os.makedirs(os.path.join(new_directory_path, trial_dirs[0]), exist_ok=True)   # directory/Hyeon_Seo/trial_0

                old_directory_path = os.path.join(directory, trial_dirs[0]) # directory/trial_0
                files = os.listdir(old_directory_path)                      # directory/trial_0/raw_data.csv    
                
                
                for file in files:         
                    shutil.move(os.path.join(old_directory_path, file), os.path.join(new_directory_path, trial_dirs[0] ,file))
                os.rmdir(old_directory_path)                               # directory/trial_0 삭제
                print(f"디렉토리를 이동했습니다: {old_directory_path} -> {os.path.join(new_directory_path, trial_dirs[0])}")
            else:
                pass
                #print(f"이미 존재하는 디렉토리입니다: {new_directory_path}")

        except Exception as e:
            print(f"디렉토리를 이동하는 도중 에러가 발생했습니다: {e}")
            if write_mode:
                log_file.write(f"{directory}\n")
                log_file.write(f"+{e}\n")
            continue
    
    log_file.close()
    print('모든 디렉토리를 이동했습니다.')

    if delete_mode:
        os.remove(f'{root_directory}/exception.txt')



def move_directory_2(root_directory, target_directory_name):
    '''
    AutoTuning
        - I.R_0150_D.R_0100_F.P_0010_S.P_0700_I.P_0400_A.T_0.070_R.I_0003
            - trial_0
        
    1. Make a new directory(Hyeon_Seo) with upper structure and move the trial_0 directory to the new directory    
    2. Merge Hyeon_Seo directories and trial_0 directories and set like this form
       (Before) -I.R_0150_D.R_0100_F.P_0010_S.P_0700_I.P_0400_A.T_0.070_R.I_0003
                    - Hyeon_Seo
                    - trial_0(or trial_1)
                        - raw_data.csv
                ----------------------------------------------------------------
                 -I.R_0150_D.R_0100_F.P_0010_S.P_0700_I.P_0400_A.T_0.070_R.I_0003
                    - Hyeon_Seo
                        -trial_0
                            - raw_data.csv
                    - trial_1

                
       (After)  -I.R_0150_D.R_0100_F.P_0010_S.P_0700_I.P_0400_A.T_0.070_R.I_0003
                    - Hyeon_Seo
                        - trial_0
                            - raw_data.csv
    '''
    total_directory = glob.glob(f'{root_directory}/*')
    total_txt_directory = glob.glob(f'{root_directory}/*.txt')
    total_csv_directory = glob.glob(f'{root_directory}/*.csv')

    for directory in total_txt_directory:
        total_directory.remove(directory)

    for directory in total_csv_directory:
        total_directory.remove(directory)

    # Extract the directory
    dir_2 = [directory for directory in total_directory if len(os.listdir(directory)) == 2]
    dir_2n = [directory for directory in dir_2 if os.listdir(directory) != ['Hyeon_Seo','Small_Joint']]
    print('Not completed : ',len(dir_2n))

    for directory in dir_2n:
        # directory/trial_0 -> directory/Hyeon_Seo/trial_0
        # directory/trial_1 -> directory/Hyeon_Seo/trial_1
        trial_dirs = os.listdir(directory)
        trial_dirs.remove('Hyeon_Seo')
        trial_dir_name = trial_dirs[0]
        empty_directory_path = os.path.join(directory,trial_dir_name)
        new_directory_path = os.path.join(directory, target_directory_name)

        files = os.listdir(empty_directory_path)
        os.makedirs(os.path.join(new_directory_path, trial_dir_name), exist_ok= True)
        for file in files:
            shutil.move(os.path.join(empty_directory_path, file), os.path.join(new_directory_path, trial_dir_name, file))
        os.rmdir(empty_directory_path)
        print(f"미완된 디렉토리를 전환했습니다: {empty_directory_path} -> {os.path.join(new_directory_path, trial_dir_name)}")

if __name__ == "__main__":
    # 디렉토리 경로와 변경할 이름 지정
    root_directory = 'D:/001.Developement of Company Work/data/AutoTuning'
    old_directory_name_list = ["Small_Joint"]
    new_directory_name_list = ["Zero_460W"]  #["Hyeon_Seo", "Small_Joint"]

    # 수정 날짜가 2024-03-06 인 파일들만 출력
    from datetime import datetime
    data_list = glob.glob(f'{root_directory}/*')
    data_list_20240306 = [data for data in data_list if datetime.fromtimestamp(os.path.getmtime(data)).strftime('%Y-%m-%d') == '2024-03-06']
    # 수정 날짜가 2024-03-06 인 파일들 중에서 '0000'으로 끝나는 디렉토리만 출력
    data_list_20240306_0000 = [data for data in data_list_20240306 if data.endswith('0000')]

    # 디렉토리 이동 함수 호출
    #move_directory(root_directory, new_directory_name_list)
    
    # 디렉토리 찾는 함수 호출
    # find_directory(root_directory, new_directory_name_list)
    # 디렉토리 삭제 함수 호출
    # remove_directory(root_directory, old_directory_name_list)

    # 디렉토리 이름 변경 함수 호출
    #rename_directory(root_directory, old_directory_name_list, new_directory_name_list)
    rename_directory_3(data_list_20240306_0000)
