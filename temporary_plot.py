import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

def plot_clustered_stacked(dfall, mean1, mean2, labels=None,  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    plt.figure(figsize=(6,4))
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xlim(-0.5, n_ind-0.2)
    axe.set_xticklabels(df.index, rotation = 0)
    plt.ylabel('NMBE %')
    plt.xlabel('POA Model')
    axe.plot(np.arange(0.125,7.125,1), mean1.values, linestyle="",marker="o",color='black',label='Mean')
    axe.plot(np.arange(0.125,7.125,1), mean2.values,linestyle="",marker="o",color='black')
    h,l = axe.get_legend_handles_labels()

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[2::-1], ['Clear','Cloud','Mean'], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.025, 0.1]) 
    axe.add_artist(l1)
    return axe, h

# create fake dataframes
df = pd.read_csv('C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Images for report/CSVs/poa.csv')
df = df.T
mean_cloud = df[[0,1,2,3]].mean(axis=1)
mean_clear = df[[5,6,7,8]].mean(axis=1)

df1 = df[[5,0]]
df1.columns = ['Cloud','Clear']
df2 = df[[6,1]]
df2.columns = ['Cloud','Clear']
df3 = df[[7,2]]
df3.columns = ['Cloud','Clear']
df4 = df[[8,3]]
df4.columns = ['Cloud','Clear']

axe, l1 = plot_clustered_stacked([df1, df2, df3, df4], mean_cloud, mean_clear, ["CO", "OR", "FL", "NM"])
# plt.plot(mean,linestyle="",marker="o",color='black',label='Mean')
#plt.legend(['Mean', 'Cloud','Clear'])
