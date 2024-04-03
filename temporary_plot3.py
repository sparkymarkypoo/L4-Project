import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

df = pd.read_csv('C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Images for report/CSVs/clear_tot.csv', index_col=0)
df = df.T
mean = df.mean(axis=1)

plt.figure(figsize=(4,4))
df.plot.bar(rot=0, width=0.8, color=CB_color_cycle, figsize=(6,4))
plt.ylabel('NMBE %')
plt.xlabel('Spectral Model')
plt.ylim(top=12.5)
plt.plot(mean,linestyle="",marker="o",color='black',label='Mean')
plt.legend()

df = pd.read_csv('C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Images for report/CSVs/cloud_tot.csv', index_col=0)
df = df.T
mean = df.mean(axis=1)

plt.figure(figsize=(4,4))
df.plot.bar(rot=0, width=0.8, color=CB_color_cycle, figsize=(6,4))
plt.ylabel('NMBE %')
plt.xlabel('Spectral Model')
plt.ylim(top=12.5)
plt.plot(mean,linestyle="",marker="o",color='black',label='Mean')
plt.legend()