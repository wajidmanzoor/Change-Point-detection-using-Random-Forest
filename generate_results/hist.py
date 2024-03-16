import os
import pandas as pd
import numpy as np
import re 
import matplotlib.pyplot as plt


RESULT_PATH = "../results"
        

file_ = open(os.path.join(RESULT_PATH,"simulation_results.txt"))
data_string = file_.read()
lines = data_string.split("\n")
new_lines = []
for line in lines:
    if line.startswith(" "):
        new_lines[-1] +=line
    else:
        new_lines.append(line)

orginal_cpts = []
predicted_cpts = []
dataset_names = []
method_names = []
for line in new_lines:
    if line=="":
        break
    else:
        temp = line.split(",[")
        cpts = temp[-2:]
        temp = temp[0].split(",")[1:3]
        dataset_names.append(temp[0])
        method_names.append(temp[1])
        orginal_cpts.append(list(map(float,re.findall(r'\d+(?:\.\d+)?',cpts[0]))))
        predicted_cpts.append(list(map(float,re.findall(r'\d+(?:\.\d+)?',cpts[1])[1:-1])))

method_names = np.array(method_names)
dataset_names = np.array(dataset_names)

def get_cpts(cpts,ind):
    return [i for i,j in zip(cpts,ind) if j]

def plot_hist(orginal_cpts,predicted_cpts,method_names,dataset_name):
    X_MARKER_KWARGS = {"marker": "x", "color": "green", "linewidth": 10, "s": 2}
    fig, axes = plt.subplots(ncols=3, nrows=2,figsize=(30,15))
    axs = axes.flat
    for ax, method_name in zip(axs,np.unique(method_names)):
        ind  = (method_names==method_name) & (dataset_names==dataset_name)
        o_cpts = get_cpts(orginal_cpts,ind)[0]
        p_cpts = get_cpts(predicted_cpts,ind)
        p_cpts = [item for sublist in p_cpts for item in sublist]
        ax.hist(p_cpts,range=(0, o_cpts[-1]),color='black',bins=int(o_cpts[-1]))
        _, ymax = ax.get_ylim()
        ax.scatter(o_cpts[1:-1], [ymax] * (len(o_cpts) - 2), **X_MARKER_KWARGS)
        ax.set_title(method_name)
    plt.show()
    
plot_hist(orginal_cpts,predicted_cpts,method_names,"Dirichlet")
plot_hist(orginal_cpts,predicted_cpts,method_names,"iris")

