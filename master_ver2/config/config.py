import argparse
import json

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser()
    # HyperParameters of System Communication
    parser.add_argument('--host_num', type = str, default = '192.168.0.23')
    parser.add_argument('--lport_num', type = int, default = 12345, help = 'local port')
    parser.add_argument('--sport_num', type = int, default = 55555, help = 'socket port to recieve data')
    
    # HyperParameters of test robot (physical property)
    parser.add_argument('--num_axes', type = int, default = 6, help = 'num-axes of robot')
    parser.add_argument('--interested_axes', type= str, default='1,2', help = 'visualize related to input axes') #1,2
    parser.add_argument('--num_inertia', type = int, default = 0, choices = [num for num in range(11)], help = 'number of inertia')

    # HyperParameters of path planning
    parser.add_argument('--num_counts', type = int, default = 21, help = 'num of repeat oscillation')
    parser.add_argument('--max_lim_speed', type = float, default = 100, help = 'max limit speed of robot')
    parser.add_argument('--acc_time_schedule', type = lambda s: [float(time) for time in s.split(',')], default = [0.0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.10], help = 'acceleration time schedule for counts')
    parser.add_argument('--dacc_time_schedule', type = lambda s: [float(time) for time in s.split(',')], default = [0.0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.10], help = 'deacceleration time schedule for counts')

    # HyperParameters of detection start time and end time of cmd pulse
    parser.add_argument('--canny_threshold', type = str, default = '10,50', help = 'find range of cmd pulse if use Edge Detection')

    # HyperParameters of Experiment Number what we are interested
    parser.add_argument('--exp_num', type = int, default =  0, help = 'experiment on the partial data of interest')
    parser.add_argument('--params_dir', type = str, default = 'remote_config.json', help = 'uploaded parameters configuration')
    parser.add_argument('--current_inertia', type = int, default = 3, help = 'current loading inertia')
    parser.add_argument('--exp_condition', type = str, default = 'meta_learning_task', help = 'type of conducting experiment') #meta_learning_task
    parser.add_argument('--reduction_ratio', type = int, default = 1, help = 'reduction ratio of motor')
    parser.add_argument('--move_radian', type = float, default = 10, help = 'move radian of motor')

    # HyperParameters of action to perform
    parser.add_argument('--generate_action', type = str2bool, default = False, help = 'generate the signal data')
    parser.add_argument('--preprocess_action', type = str2bool, default = True, help = 'preprocess of the signal data')
    parser.add_argument('--summarize_action', type =str2bool, default = False, help = 'summarize of the signal data')
    parser.add_argument('--plot_action', type = str2bool, default = False, help = 'visualize the signal data in given time range')
    parser.add_argument('--plot_ind', type = str2bool, default = False, help = 'plot pulse individually')
    parser.add_argument('--vis_mode', type = str, default = 'heatmap'  ,choices= ['pulse', 'heatmap'])
    
    
    # Make Namespace of argument parser
    config = parser.parse_args()

    # HWDEF, JUDEF 
    if config.exp_condition in ['data_generation_Hyeon_Seo', 'data_geneation_Small_Joint','data_generation_Zero_460W',  'meta_learning_task', 'dummy_finner_experiment']:
        HWDEF_name = 'HWDEF_20240213_comparison_test.json'
    else:
        HWDEF_name = 'HWDEF.json'
    with open(f"./config/{HWDEF_name}", "r") as file:
        num_axes = json.load(file)['system']['n_axes'][0]

    config.num_axes = num_axes

    # Set max_limit_speed(%) and acc_time(dacc_time)

    return config

if __name__ == '__main__':
    args = get_parser()

    print(args)