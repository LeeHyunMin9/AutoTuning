from config.config import get_parser
import subprocess
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s", datefmt="[%X]")

def log_stderr(program: str, line: str) -> None:
    logger.debug("%s: %s", program, line.rstrip())


if __name__ == '__main__':

    args = get_parser()
    # 현재는 Unit Step Function 을 통한 실험만 진행중
    
    if args.generate_action:
        '''
        전달 인자 :    
                      exp_num         : 반복 횟수
                      current_inertia : 현재 이너샤 세팅
                      params_dir      : 파라미터 업로드 디렉토리
                      exp_condition   : 어떤 실험을 진행 했는지?
                                        data_generation : 어떤 데이터를 생성할 지
                                        xxxx_experiment : 실험 조건에 대한 명시
                      interested_axes : 원하는 축의 번호들
                      degree          : 움직이는 각도
                      reduction_ratio : 감속비
                      full_axis       : 초기 축의 개수
                      
        '''
        cmd = ["python","generate_data.py", str(args.exp_num), str(args.current_inertia), args.params_dir, args.exp_condition,
                            args.interested_axes, str(args.move_radian), str(args.reduction_ratio), str(args.num_axes)]
        
        debug = False

        # 1. debugging 단계로 진입
        if debug:
            with subprocess.Popen(cmd) as proc:
                stdout, stderr = proc.communicate()
            result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
            print(result)

        else:
            with subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr= subprocess.PIPE, text = True) as proc:
                for line in proc.stderr:
                    log_stderr(cmd[0], line)
                stdout, stderr = proc.communicate()
            result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
            print(result)

    if args.preprocess_action:
        '''
        전달 인자 : 원하는 축에 대한 데이터 추출 이후 관련 있는 열에 대해서 가공
                      exp_num         :  반복 횟수
                      num_axes        :  축의 개수
                      interested_axes :  원하는 축의 번호들
                      exp_condition   :  어떤 실험을 진행 했는지? 
                      

        출력값 :   원하는 축에 대한 데이터 요약값
                    ------------From Pulse Data----------------
                    + overshoot_percentage
                    + settling_time
                    + rise_time
                    + peak_time
                    + peak_value
                    + steady_state_value
                    + steady_state_error
                    + RMSE values
                    ------------From Torque Data----------------
                    + Number of Frequency
                    + SNR(cutoff frequency  )
                    + Clusters 
        '''
        cmd = ["python","preprocess_data.py", str(args.exp_num), str(args.num_axes), args.interested_axes, args.exp_condition]
        debug = True
        # 1. debugging 단계로 진입
        if debug:
            with subprocess.Popen(cmd) as proc:
                stdout, stderr = proc.communicate()
            result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
            print(result)
        # 2. debugging 필요 없을 때
        else:
            with subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr= subprocess.PIPE, text = True) as proc:
                for line in proc.stderr:
                    log_stderr(cmd[0], line)
                stdout, stderr = proc.communicate()
            result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
            print(result)
        
    
    if args.plot_action:
        # 데이터 폴더로부터 특성값을 gathering 해서 추가적인 작업을 진행한다.
        subprocess.Popen(["python","plot_tools.py", args.exp_condition])

        if args.plot_action and not args.train_load_action:    
            # 데이터 폴더로부터 특성값을 gathering 해서 추가적인 작업을 진행한다.
            # 그렸었던 plot : outlier.py(3d plot 에서 이상치 표시), 
            '''
            전달 인자 : 
            '''
        
            subprocess.Popen(["python","plot_tools.py", args.exp_condition])

    if args.plot_action and args.train_load_action:
        # 데이터 폴더로부터 특성값을 gathering 해서 추가적인 작업을 진행한다.
        # 그렸었던 plot : outlier.py(3d plot 에서 이상치 표시), 
        '''
        전달 인자 : 
        '''
        print('HI')
        #subprocess.Popen(["python","plot_tools.py", args.exp_condition, args.train_load_action])