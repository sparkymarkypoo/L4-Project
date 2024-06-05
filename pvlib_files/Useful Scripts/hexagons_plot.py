
# Demonstration of creating hexagon grid over census map and plotting it

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union
import numpy as np
import os
cwd = os.path.dirname(os.path.dirname(os.getcwd()))

# Read shapefile
gdf = gpd.read_file(os.path.join(cwd, 'Data\cb_2018_us_state_500k\cb_2018_us_state_500k.shp'))

# Get US without territories / Alaska + Hawaii
usa = gdf[~gdf.NAME.isin(["Hawaii", "Alaska", "American Samoa", 
                         "United States Virgin Islands", "Guam",
                         "Commonwealth of the Northern Mariana Islands",
                         "Puerto Rico"])]

# Convert to EPSG 4326 for compatibility with H3 Hexagons
usa = usa.to_crs(epsg=4326)
# Get union of the shape (whole US)
union_poly = unary_union(usa.geometry)

#import h3pandas
resolution = 3
hexagons = usa.h3.polyfill_resample(resolution)
hexagons['hello'] = np.random.randint(1, 10, hexagons.shape[0])

# Centres
hexagons['LAT'] = hexagons.centroid.map(lambda p: p.y)
hexagons['LONG'] = hexagons.centroid.map(lambda p: p.x)

# Plot
fig, ax = plt.subplots(figsize=(15, 15))
usa.plot(ax=ax, facecolor="none", edgecolor='black', linewidth=1)
hexagons.plot(ax=ax, column='hello', alpha=0.7, cmap='coolwarm_r')
for i in range(len(hexagons)):
    plt.text(hexagons.LONG[i],hexagons.LAT[i],'x',size=10)
plt.show()