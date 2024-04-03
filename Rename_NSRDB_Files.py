import os
import pandas as pd
from shapely.geometry import Point
from US_map import hexagons_map
hexagons = hexagons_map()

folder = 'D:\Hexagon_downloads'
arr = os.listdir(folder)

for a in arr:
    
    oldname = f'{folder}/{a}'
    
    metadata = pd.read_csv(oldname, nrows=1)
    data = pd.read_csv(oldname, skiprows=2, nrows=1, usecols=['Year'])
    lat = float(metadata['Latitude'])
    long = float(metadata['Longitude'])
    
    point = Point(long, lat)
    gon = ' '.join(map(str, hexagons.index[point.within(hexagons.geometry)]))
    
    newname = f'{folder}/{gon}_lat{lat}_long{long}_nsrdb.csv'
    os.rename(oldname, newname)