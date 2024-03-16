import os
import pandas as pd
import numpy as np

RESULT_PATH = "../results/"

TABLES_PATH = os.path.join(RESULT_PATH,'tables')
TEX_PATH = os.path.join(RESULT_PATH,'tex')

if not os.path.exists(TEX_PATH):
    os.mkdir(TEX_PATH)
    
try:
    mean_rand_df =  pd.read_csv(os.path.join(TABLES_PATH,"mean_rand_index.csv"),index_col=0)
    std_rand_df = pd.read_csv(os.path.join(TABLES_PATH,"std_rand_index.csv"),index_col=0)
    mean_time_df = pd.read_csv(os.path.join(TABLES_PATH,"mean_time.csv"),index_col=0)
except:
    ValueError ("CSV not found")
    
table_start = r"""
\begin{table}[!ht]
\centering
{\rowcolors{3}{green!80!yellow!50}{green!70!yellow!40}
"""


def add_header(table_start, cols,title):
    table_start += r"""\begin{tabular}{ |p{2cm}|""" 
    table_start += "p{1.5cm}|"*(cols-1)
    table_start += r"""}
\hline
\multicolumn{"""
    table_start += str(cols)
    table_start += r"""}{|c|}{"""
    table_start += title
    table_start += r"""} \\
\hline
"""
    return table_start

def add_footer(table_start,caption="Hah"):
    table_start += r"""\hline
\end{tabular}}
\\
\caption{"""
    table_start +=   caption
    table_start +=r"""}
\vspace{10pt}
\label{tab:yourlabel}
\end{table}"""
    return table_start

def add_rand_table(table_start,mean_df,std_df,start_col,end_col):
    table_start +=r"""
\hline
"""
    for col in mean_df.columns[start_col:end_col]:
        table_start += " & "
        table_start += col
    table_start += " & Average & Worst "
    table_start += r""" \\
\hline
"""
    table_start += r"""\hline
"""
    for ind in mean_df.index:
        mean_row = mean_df.loc[ind][start_col:end_col]
        std_row = std_df.loc[ind][start_col:end_col]
        table_start += f"{ind}"
        for mean,std in zip(mean_row,std_row):
            table_start += f" & {mean} ({np.round(std,5)})"
        table_start += f"& {np.round(mean_row.mean(),2)} & {mean_row.min()}"
        table_start += r""" \\
"""
    return table_start

def add_time_table(table_start,time_df,start_col,end_col):
    table_start +=r"""
\hline
"""
    for col in time_df.columns[start_col:end_col]:
        table_start += " & "
        table_start += col
    table_start += r""" \\
\hline
"""
    table_start += r"""\hline
"""
    for ind in time_df.index:
        row = time_df.loc[ind][start_col:end_col]
        table_start += f"{ind}"
        for time_ in row:
            table_start += f" & {time_}"
        table_start += r""" \\
"""
    return table_start






generated_dataset_tex = add_header(table_start,7,"Average Adjusted Rand Index")
generated_dataset_tex = add_rand_table(generated_dataset_tex,mean_rand_df,std_rand_df,0,4)
generated_dataset_tex = add_footer(generated_dataset_tex)

file_ = open(os.path.join(TEX_PATH,"table_main_generated_datasets.tex"),"w")
file_.write(generated_dataset_tex)
file_.close()

other_dataset_tex = add_header(table_start,7,"Average Adjusted Rand Index")
other_dataset_tex = add_rand_table(other_dataset_tex,mean_rand_df,std_rand_df,4,8)
other_dataset_tex = add_footer(other_dataset_tex)

file_ = open(os.path.join(TEX_PATH,"table_main_other_datasets.tex"),"w")
file_.write(other_dataset_tex)
file_.close()

time_gen_tex = add_header(table_start,5,"Average Run Time (seconds)")
time_gen_tex = add_time_table(time_gen_tex,mean_time_df,0,4)
time_gen_tex = add_footer(time_gen_tex,"hah")

file_ = open(os.path.join(TEX_PATH,"table_time_generated_datasets.tex"),"w")
file_.write(time_gen_tex)
file_.close()

time_other_tex = add_header(table_start,5,"Average Run Time (seconds)")
time_other_tex = add_time_table(time_other_tex,mean_time_df,4,8)
time_other_tex = add_footer(time_other_tex,"hah")

file_ = open(os.path.join(TEX_PATH,"table_time_other_datasets.tex"),"w")
file_.write(time_other_tex)
file_.close()