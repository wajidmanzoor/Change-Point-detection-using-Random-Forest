import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def plot_results(df,title,method_names,n_segments_list):
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'height_ratios': [9, 1]})
    ax[1][1].set_position([ax[1][0].get_position().x0, ax[1][0].get_position().y0,
                        ax[1][1].get_position().x1 - ax[1][0].get_position().x0,
                        ax[1][0].get_position().height])
    ax[1][0].remove()
    fig.suptitle(title)
    colors = cm.rainbow(np.linspace(0, 1, len(method_names)))
    for j,segment in enumerate(n_segments_list): 
        for i,method_name in enumerate(method_names):
            ind = (df['method']==method_name) & (df['segments']==segment)
            temp = df[ind]
            ax[0][j].plot(temp['observations'],temp['rand_index'],label=method_name,color=colors[i])
            ax[0][j].set_title("segments: "+str(segment))
            ax[0][j].set_xlabel("Sample Size")
            ax[0][j].set_ylabel("Rand Index")
    for i, method_name in enumerate(method_names):
        ax[1][1].plot([], [], label=method_name, color=colors[i])

    ax[1][1].legend(loc='center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=len(method_names))

    ax[1][1].xaxis.set_visible(False)
    ax[1][1].yaxis.set_visible(False)

    ax[1][1].spines['top'].set_visible(False)
    ax[1][1].spines['right'].set_visible(False)
    ax[1][1].spines['bottom'].set_visible(False)
    ax[1][1].spines['left'].set_visible(False)


    plt.show()
    
col_names=["dataset","method","rand_index","time",'segments',"observations"]
method_names = ['Change in Mean', 'change KNN', 'KCP-linear', 'KCP-rbf','MultiRank','Change Forest']
n_segments_list = [20, 80]

df = pd.read_csv("../results/dirichlet_results_varying_segments.txt",names=col_names)
plot_results(df,"Dirichlet",method_names,n_segments_list)

df = pd.read_csv("../results/Wine_with_noise_results_varying_segments.txt",names=col_names)

plot_results(df,"Wine",method_names,n_segments_list)