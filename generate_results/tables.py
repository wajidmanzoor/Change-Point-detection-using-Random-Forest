import os
import pandas as pd
import numpy as np


RESULT_PATH = "../results/"
TABLES_PATH = os.path.join(RESULT_PATH,'tables')
        

file_ = open(os.path.join(RESULT_PATH,"simulation_results.txt"))
data_string = file_.read()
lines = data_string.split("\n")
new_lines = []
for line in lines:
    if line.startswith(" "):
        new_lines[-1] +=line
    else:
        new_lines.append(line)
data = []
for line in new_lines:
    if line =="":
        break
    data.append(line.split(",")[:5])

df = pd.DataFrame(data,columns=['simulation','dataset_name','method_name','rand_index','time'])

df['simulation'] = df['simulation'].astype(int)
df['rand_index'] = df['rand_index'].astype(float)
df['time'] = df['time'].astype(float)

dataset_names = ['CIM', 'CIV','CIC', 'Dirichlet', 'abalone', 'glass', 'iris','wine']
method_names = ['Change in Mean', 'change KNN', 'KCP-linear', 'KCP-rbf','MultiRank','Change Forest']

mean_rand_data = []
std_rand_data = []
mean_time = []
for method_name in method_names:
    temp1 = []
    temp2 = []
    temp3 = []
    for dataset_name in dataset_names:
        ind = (df['dataset_name']==dataset_name) &(df['method_name']==method_name)
        temp1.append(np.round(df[ind]['rand_index'].mean(),2))
        temp2.append(df[ind]['rand_index'].std())
        temp3.append(np.round(df[ind]['time'].mean(),2))
    mean_rand_data.append(temp1)
    std_rand_data.append(temp2)
    mean_time.append(temp3)
mean_rand_df = pd.DataFrame(mean_rand_data,columns=dataset_names,index=method_names)
std_rand_df = pd.DataFrame(std_rand_data,columns=dataset_names,index=method_names)
mean_time_df = pd.DataFrame(mean_time,columns=dataset_names,index=method_names)

mean_rand_df.to_csv(os.path.join(TABLES_PATH,"mean_rand_index.csv"))
std_rand_df.to_csv(os.path.join(TABLES_PATH,"std_rand_index.csv"))
mean_time_df.to_csv(os.path.join(TABLES_PATH,"mean_time.csv"))

