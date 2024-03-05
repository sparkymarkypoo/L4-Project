import pandas as pd
import pvlib

# # solar farms list from https://eerscmap.usgs.gov/uspvdb/data/
# sf = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/FINAL/US_Solar_Farms.csv')

# sf_max = sf.sort_values('p_cap_dc', ascending=False).drop_duplicates(['p_state'])
# sf_max.index = sf_max.p_state

# sf_max['opt_tilt'] = 0
# for loc in sf_max.index:
#     start = 2015
#     end = start
#     lat = sf_max.ylat.loc[loc]
#     long = sf_max.xlong.loc[loc]
    
#     data, inputs, metadata = pvlib.iotools.get_pvgis_hourly(latitude=lat, longitude=long, start=start, end=end,
#                                           raddatabase=None, components=True, surface_tilt=0, surface_azimuth=180,
#                                           outputformat='json', usehorizon=True, userhorizon=None, pvcalculation=False,
#                                           peakpower=None, pvtechchoice='crystSi', mountingplace='free', loss=0,
#                                           trackingtype=0, optimal_surface_tilt=True, optimalangles=False,
#                                           url='https://re.jrc.ec.europa.eu/api/', map_variables=True, timeout=30)
    
#     sf_max['opt_tilt'].loc[loc] = inputs['mounting_system']['fixed']['slope']['value']
    
# sf_max.to_csv('C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/FINAL/US_Solar_Farms_Cropped.csv')



from US_map import state_map
coords = state_map()
coords['LAT'] = coords.centroid.map(lambda p: p.y)
coords['LONG'] = coords.centroid.map(lambda p: p.x)

coords['opt_tilt'] = 0
for loc in coords.index:
    start = 2015
    end = start
    lat = coords.LAT.loc[loc]
    long = coords.LONG.loc[loc]
    
    data, inputs, metadata = pvlib.iotools.get_pvgis_hourly(latitude=lat, longitude=long, start=start, end=end,
                                          raddatabase=None, components=True, surface_tilt=0, surface_azimuth=180,
                                          outputformat='json', usehorizon=True, userhorizon=None, pvcalculation=False,
                                          peakpower=None, pvtechchoice='crystSi', mountingplace='free', loss=0,
                                          trackingtype=0, optimal_surface_tilt=True, optimalangles=False,
                                          url='https://re.jrc.ec.europa.eu/api/', map_variables=True, timeout=30)
    
    coords['opt_tilt'].loc[loc] = inputs['mounting_system']['fixed']['slope']['value']
   
new_coords = coords[['STATEFP','LAT','LONG','opt_tilt']]
new_coords.to_csv('C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/FINAL/US_State_Centres.csv')