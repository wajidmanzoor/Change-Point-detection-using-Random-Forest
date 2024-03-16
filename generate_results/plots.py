import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

RESULT_PATH = "../results"
FIGURE_PATH = os.path.join(RESULT_PATH,'figures')

def get_segment_data(df,method_names,segment):
    rand_index =[]
    time_ = []
    n_observations = np.unique(df['observations'])
    for method_name in method_names:
        temp1 = []
        temp2 = []
        for observation in n_observations:
            ind = (df['method']==method_name)&(df['segments']==segment) & (df['observations']==observation)
            temp1.append(df[ind]['rand_index'].mean())
            temp2.append(df[ind]['time'].mean())
        rand_index.append(temp1)
        time_.append(temp2)
    return rand_index,time_, n_observations
def get_data(df,method_names,segment_list=[20,80]):
    rand_scores = []
    runtimes = []
    for segment in segment_list:
        score,time_,observation = get_segment_data(df,method_names,segment)
        rand_scores.append(score)
        runtimes.append(time_)
    return rand_scores,runtimes,observation

def plot_varying_samplesize_results(x_data,y_data,title,y_axis_name,x_axis_name,path):
        fig, ax = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'height_ratios': [9, 1]})
        ax[1][1].set_position([ax[1][0].get_position().x0, ax[1][0].get_position().y0,
                        ax[1][1].get_position().x1 - ax[1][0].get_position().x0,
                        ax[1][0].get_position().height])
        ax[1][0].remove()
        fig.suptitle(title)
        colors = cm.rainbow(np.linspace(0, 1, len(method_names)))
        for i,(segment,y_segment) in enumerate(zip(n_segments_list,y_data)):
                for j,(Y, method_name) in enumerate(zip(y_segment,method_names)):
                        ax[0][i].plot(x_data,Y,label=method_name,color=colors[j])
                        ax[0][i].set_title("segments: "+str(segment))
                        ax[0][i].set_xlabel(x_axis_name)
                        ax[0][i].set_ylabel(y_axis_name)
        for i, method_name in enumerate(method_names):
                ax[1][1].plot([], [], label=method_name, color=colors[i])

        ax[1][1].legend(loc='center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=len(method_names))

        ax[1][1].xaxis.set_visible(False)
        ax[1][1].yaxis.set_visible(False)

        ax[1][1].spines['top'].set_visible(False)
        ax[1][1].spines['right'].set_visible(False)
        ax[1][1].spines['bottom'].set_visible(False)
        ax[1][1].spines['left'].set_visible(False)
        filename =  f"{title}_effect_of_varing_sample_size_on_{y_axis_name}.png"
        plt.savefig(os.path.join(path,filename))
        plt.show()

        

col_names=["seed","dataset","method","rand_index","time",'segments',"observations"]
method_names = ['Change in Mean', 'change KNN', 'KCP-linear', 'KCP-rbf','MultiRank','Change Forest']
n_segments_list = [20, 80]

df  = pd.read_csv(os.path.join(RESULT_PATH,"dirichlet_results_varying_segments (2).txt"),names=col_names)
if len(df)%132!=0:
    df= df[df['seed']!=df['seed'][len(df)-1]]
    
rand_scores,runtimes,observation = get_data(df,method_names,segment_list=[20,80])
plot_varying_samplesize_results(observation,rand_scores,"Dirichlet","Adjusted Rand Index","Sample Size",FIGURE_PATH)
plot_varying_samplesize_results(observation,runtimes,"Dirichlet","Runtime","Sample Size",FIGURE_PATH)

