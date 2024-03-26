# Utility Functions

import time
import pandas as pd
import numpy as np
import os

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        # if args and kwargs:
        #     print("WorkingTime[{}] with Args[{}] and Kwargs[{}]: {:.2f} sec".format(original_fn.__name__, args, kwargs, end_time-start_time))
        # elif args and not kwargs:
        #     print("WorkingTime[{}] with Args[{}]: {:.2f} sec".format(original_fn.__name__, args, end_time-start_time))
        # elif kwargs and not args:
        #     print("WorkingTime[{}] with Args[{}] and Kwargs[{}]: {:.2f} sec".format(original_fn.__name__, kwargs, end_time-start_time))
        # else:
        print("WorkingTime[{}] : {:.2f} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn



def unit_from_name(name):
    return '%' if name in ['I.R', 'D.R'] else \
           rf'rad/$s^2$' if name in ['F.P','S.P','I.P'] else \
           rf's' if name in ['A.T'] else \
           rf'x'

def UpperCase_of_Text(text, standard = " "):
    '''
    문자열의 공백 기준으로 단어의 첫번째를 대문자로 바꾸기
    'my name is HyunMin' -> 'My Name Is HyunMin'
    '''
    Split_Text = text.split(standard)

    Result_Text = ""

    for inner_text in Split_Text:

        inner_text = inner_text.capitalize()

        if Result_Text == "":
            Result_Text = inner_text
        else:
            Result_Text = Result_Text + standard + inner_text

    return Result_Text

def index_from_state_dict(state_dict_directory):
    target_loss = float('inf')
    target_index = 0
    
    for directory in state_dict_directory:
        fold, epoch = directory.split('\\')[-1].split('_')[1], int(directory.split('\\')[-1].split('_')[3])
        
        loss_train = pd.read_csv( os.path.join(  os.path.dirname(directory.split('\\')[0]), 'losses', f'fold_{fold}_train.csv')).iloc[epoch][1]
        loss_val = pd.read_csv(os.path.join(  os.path.dirname(directory.split('\\')[0]), 'losses', f'fold_{fold}_val.csv')).iloc[epoch][1]

        if np.sqrt(loss_train **2 + loss_val **2) < target_loss:
            target_loss = np.sqrt(loss_train **2 + loss_val **2)
            target_index = int(fold)
    return target_index

