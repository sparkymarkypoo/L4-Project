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
 
   
import pandas as pd
import time

coords = pd.read_csv('C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/FINAL/US_State_Centres.csv', index_col='STATEFP')
for loc in coords.index:
    for y in range(2021,2022):
        while True:
            response = spectral_on_demand_API(lat=coords['LAT'].loc[loc], long=coords['LONG'].loc[loc], year=y, tilt=coords['opt_tilt'].loc[loc])
            if response.ok == True:
                print('Passed:', loc, y)
                time.sleep(2)
                break
            else:
                print('Failed:', loc, y)
                time.sleep(60*2)

# sf = pd.read_csv('C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/FINAL/US_Solar_Farms_Cropped.csv', index_col='p_state')
# for loc in sf.index:
#     for y in range(2018,2022):
#         while True:
#             response = spectral_on_demand_API(lat=sf['ylat'].loc[loc], long=sf['xlong'].loc[loc], year=y, tilt=sf['opt_tilt'].loc[loc])
#             if response.ok == True:
#                 print('Passed:', loc, y)
#                 time.sleep(2)
#                 break
#             else:
#                 print('Failed:', loc, y)
#                 time.sleep(60*2)