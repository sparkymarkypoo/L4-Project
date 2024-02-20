import os
import pandas as pd
from shapely.geometry import Point
from US_map import state_map
coords = state_map()

folder = 'D:/NSRDB_Data'
arr = os.listdir(folder)

for a in arr:
    
    oldname = f'{folder}/{a}'
    
    metadata = pd.read_csv(oldname, nrows=1)
    data = pd.read_csv(oldname, skiprows=2, nrows=1, usecols=['Year'])
    lat = float(metadata['Latitude'])
    long = float(metadata['Longitude'])
    year = data['Year'].iloc[0]
    
    point = Point(long, lat)
    state = ' '.join(map(str, coords.index[point.within(coords.geometry)]))
    
    newname = f'{folder}/state{state}_lat{lat}_long{long}_year{year}_nsrdb.csv'
    os.rename(oldname, newname)