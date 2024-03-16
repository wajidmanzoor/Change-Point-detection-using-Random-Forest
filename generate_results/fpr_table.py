import os
import pandas as pd
import numpy as np
import re 

RESULT_PATH = "../results"
TABLES_PATH = os.path.join(RESULT_PATH,'tables')


file_ = open(os.path.join(RESULT_PATH,"simulation_false_positive_rate (2).txt"))
data_string = file_.read()
lines = data_string.split("\n")
new_lines = []
for line in lines:
    if line.startswith(" "):
        new_lines[-1] +=line
    else:
        new_lines.append(line)
        
seed =[]
predicted_cpts = []
dataset_names = []
method_names = []
num_cpt = []
for line in new_lines:
    if line=="":
        break
    else:
        temp = line.split(",[")
        cpts = temp[-1]
        temp = temp[0].split(",")
        dataset_names.append(temp[1])
        method_names.append(temp[2])
        seed.append(int(temp[0]))
        num_cpt.append(float(temp[4]))
        predicted_cpts.append(list(map(float,re.findall(r'\d+(?:\.\d+)?',cpts[1])[1:-1])))

method_names = np.array(method_names)
dataset_names = np.array(dataset_names)
df = pd.DataFrame()
df['seed'] = seed
df['dataset_name'] = dataset_names
df['method_name'] = method_names
df['false_postive'] = np.array(num_cpt)-2
if len(df)%1200 !=0:
    df = df[df['seed']!=df['seed'][len(df)-1]]
u_dataset_names = ['CIM', 'CIV','CIC', 'Dirichlet', 'abalone', 'glass', 'iris','wine']
u_method_names = ['Change in Mean', 'change KNN', 'KCP-linear', 'KCP-rbf','MultiRank','Change Forest']
false_positive = []
for method_name in u_method_names:
    temp1 = []
    for dataset_name in u_dataset_names:
        ind = (df['dataset_name']==dataset_name) &(df['method_name']==method_name)
        temp = df[ind]
        temp1.append(temp['seed'][temp['false_postive']>0].count())
    false_positive.append(temp1)
out_df = pd.DataFrame(np.round(np.array(false_positive)/450,2),columns=u_dataset_names,index=u_method_names)
out_df.to_csv(os.path.join(TABLES_PATH,"FPR_450_simulations.csv"))