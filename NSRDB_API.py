def PSMv3_API(latitude, longitude, year, interval):
    
    import pvlib
    
    attributes=('air_temperature', 'dhi', 'dni', 'ghi',
    'surface_albedo', 'wind_speed', 'cloud_type',
    'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi')
    #,'dew_point', 'surface_pressure'
    
    data, metadata = pvlib.iotools.get_psm3(latitude=latitude, longitude=longitude, names=f'{year}', 
                                            attributes=attributes, interval=interval,
                                            api_key='hQ8W1ZA3mLtMmrP7ee45addpSTwyh1Kid4rurDRc',
                                            email='mark.r.salkeld@gmail.com', leap_day=False,
                                            full_name='Mark Salkeld', affiliation='pvlib python', timeout=30,
                                            map_variables=False)
    
    return data

def spectral_on_demand_API(lat, long, year, tilt):
     import requests
     
     api_key ='hQ8W1ZA3mLtMmrP7ee45addpSTwyh1Kid4rurDRc'
     first_name = 'Mark'
     surname = 'Salkeld'
     email ='mark.r.salkeld@gmail.com'
     affiliation ='DurhamUniversity'
     equipment = 'fixed_tilt'
     azimuth = 180
     tilt = int(tilt)

     response = requests.get(f'http://developer.nrel.gov/api/nsrdb_api/solar/spectral_ondemand_download.json?api_key={api_key}&wkt=POINT({long}%20{lat})&names={year}&full_name={first_name}%20{surname}&email={email}&affiliation={affiliation}&mailing_list=false&reason=test&equipment={equipment}&tilt={tilt}&angle={azimuth}')

     return response
 
   
# import pandas as pd
# import time

# coords = pd.read_csv('C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/FINAL/US_State_Centres.csv', index_col='STATEFP')
# for loc in coords.index:
#     for y in range(2021,2022):
#         while True:
#             response = spectral_on_demand_API(lat=coords['LAT'].loc[loc], long=coords['LONG'].loc[loc],
#                                               year=y, tilt=coords['opt_tilt'].loc[loc])
#             if response.ok == True:
#                 print('Passed:', loc, y)
#                 time.sleep(2)
#                 break
#             else:
#                 print('Failed:', loc, y)
#                 time.sleep(60*2)

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union
import numpy as np
import time
import pvlib

# Read USA map shapefile
gdf = gpd.read_file("C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Grid/cb_2018_us_state_500k/cb_2018_us_state_500k.shp")

# Get USA without territories / Alaska + Hawaii
usa = gdf[~gdf.NAME.isin(["Hawaii", "Alaska", "American Samoa", 
                         "United States Virgin Islands", "Guam",
                         "Commonwealth of the Northern Mariana Islands",
                         "Puerto Rico"])]

# Convert to EPSG 4326 for compatibility with H3 Hexagons
usa = usa.to_crs(epsg=4326)

# Get union of the shape (whole USA)
union_poly = unary_union(usa.geometry)

import h3pandas
resolution = 3
hexagons = usa.h3.polyfill_resample(resolution)

# Get hexagon centres
hexagons['LAT'] = hexagons.centroid.map(lambda p: p.y)
hexagons['LONG'] = hexagons.centroid.map(lambda p: p.x)

# Get optimum angles
import pandas as pd
gons = pd.read_csv('C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/hexagons.csv', index_col=0)
hexagons['opt_tilt'] = gons['opt_tilt']
# hexagons['opt_tilt'] = 0
# for loc in hexagons.index:
#     start = 2015
#     end = start
#     lat = hexagons.LAT.loc[loc]
#     long = hexagons.LONG.loc[loc]             
#     data, inputs, metadata = pvlib.iotools.get_pvgis_hourly(latitude=lat, longitude=long, start=start, end=end,
#                                           raddatabase=None, components=True, surface_tilt=0, surface_azimuth=180,
#                                           outputformat='json', usehorizon=True, userhorizon=None, pvcalculation=False,
#                                           peakpower=None, pvtechchoice='crystSi', mountingplace='free', loss=0,
#                                           trackingtype=0, optimal_surface_tilt=True, optimalangles=False,
#                                           url='https://re.jrc.ec.europa.eu/api/', map_variables=True, timeout=30)   
#     hexagons['opt_tilt'].loc[loc] = inputs['mounting_system']['fixed']['slope']['value']


# Run Spectral on Demand API
hexagons = hexagons.loc['832643fffffffff':]
for loc in hexagons.index:
          while True: # prevents network error
              response = spectral_on_demand_API(lat=hexagons['LAT'].loc[loc], long=hexagons['LONG'].loc[loc],
                                                year=2021, tilt=hexagons['opt_tilt'].loc[loc])             
              if response.ok == True:
                print('Passed:', loc)
                time.sleep(2)
                break
              else:
                print('Failed:', loc)
                time.sleep(60*2)