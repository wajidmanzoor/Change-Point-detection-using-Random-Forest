import os
import pandas as pd
import numpy as np

RESULT_PATH = "../results/"

TABLES_PATH = os.path.join(RESULT_PATH,'tables')
TEX_PATH = os.path.join(RESULT_PATH,'tex')

if not os.path.exists(TEX_PATH):
    os.mkdir(TEX_PATH)
    
try:
    fpr_df = pd.read_csv(os.path.join(TABLES_PATH,"FPR_450_simulations.csv"),index_col=0)
except:
    ValueError ("CSV not found")
    
table_start = r"""
\begin{table}[!ht]
\centering
{\rowcolors{3}{green!80!yellow!50}{green!70!yellow!40}
"""

def add_header(table_start, cols,title):
    table_start += r"""\begin{tabular}{ |p{2cm}|""" 
    table_start += "p{0.9cm}|"*(cols-1)
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

def add_rand_table(table_start,mean_df,start_col,end_col):
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
        table_start += f"{ind}"
        for mean in mean_row:
            table_start += f" & {mean}"
        table_start += f"& {np.round(mean_row.mean(),2)} & {mean_row.max()}"
        table_start += r""" \\
"""
    return table_start
fpr = add_header(table_start,11,"False Positive Rate")
fpr = add_rand_table(fpr,fpr_df,0,8)
fpr = add_footer(fpr,"add label here")

file_ = open(os.path.join(TEX_PATH,"FPR_simulations.tex"),'w')
file_.write(fpr)
file_.close()