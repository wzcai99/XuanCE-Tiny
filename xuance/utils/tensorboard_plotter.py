from ast import arg
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def smooth_value(data_list,coff=0.9):
    result = []
    last = data_list[0]
    for d in data_list:
        result.append(coff*last + (1-coff)*d)
        last = coff*last + (1-coff)*d
    return result
def smooth_step(data_list,coff=1000):
    return list(np.array(data_list).astype(np.int32)//coff * float(coff))

sns.set_theme()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",type=str,default="HalfCheetah")
    parser.add_argument("--log_dir",type=str,default="./logs/")
    parser.add_argument("--key_word",type=str,default="rewards-steps")
    parser.add_argument("--y_smooth",type=float,default=0.9)
    parser.add_argument("--x_smooth",type=float,default=1000)
    args = parser.parse_known_args()[0]
    fig = plt.figure(dpi=200)
    for env_dir in os.listdir(args.log_dir):
        if args.env_name in env_dir:
            for algo_name in os.listdir(os.path.join(args.log_dir,env_dir)):
                algo_path = os.path.join(args.log_dir,env_dir,algo_name)
                algo_data_x = []
                algo_data_y = []
                env_num = 0
                for log_data_dir in os.listdir(algo_path):
                    log_data_path = os.path.join(algo_path,log_data_dir)
                    if args.key_word in log_data_path: 
                        log_data_path = os.path.join(log_data_path,os.listdir(log_data_path)[0])
                        log_data = event_accumulator.EventAccumulator(log_data_path)
                        log_data.Reload()
                        algo_data_x.extend(smooth_step([data.step for data in log_data.scalars.Items(args.key_word)],coff=args.x_smooth))
                        algo_data_y.extend(smooth_value([data.value for data in log_data.scalars.Items(args.key_word)],coff=args.y_smooth))
                        env_num += 1
                        
                plot_data_x = np.array(algo_data_x)[:,np.newaxis] * env_num
                plot_data_y = np.array(algo_data_y)[:,np.newaxis]
                plot_data = np.concatenate((plot_data_x,plot_data_y))
                pd_data = pd.DataFrame(np.concatenate((plot_data_x,plot_data_y),axis=-1),columns=['steps','rewards'])
                sns.lineplot(pd_data,x="steps",y="rewards",legend='brief',label=algo_name)
    
    plt.title(args.env_name)
    plt.ticklabel_format(style='sci',scilimits=(0,1),axis='x')
    plt.show()
                    

                
    
