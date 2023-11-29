# Import pvlib stuff
import pvlib
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# Import general stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import data
from Data.open_data import NRELdata

celltype = 'monoSi'
pdc0 = 400
v_mp = 44.1
i_mp = 9.08
v_oc = 53.4
i_sc = 9.60
alpha_sc = 0.0005 * i_sc
beta_voc = -0.0029 * v_oc
gamma_pdc = -0.0037
cells_in_series = 6*27
temp_ref = 25

location = Location(latitude=36.626,longitude=0,#-116.018
                    name='Desert Rock',altitude=1000 #,tz='US/Pacific'
                    )
surface_azimuth=180
surface_tilt=37

start = np.datetime64('2022-05-26T00:00')
end = np.datetime64('2022-05-30T00:00')

# Get NREL (FARMS) data
file = r'C:\Users\mark\Documents\L4-Project-Code\Data\Desert Rock\NREL_2022_DRA_5min.csv'
interval='5m'
year, FARMS_ghi, FARMS_dhi, FARMS_dni, cloud_type, wind, air_temp = NRELdata(start,end)
FARMS_time = np.arange(start, end+5, dtype=f'datetime64[{interval}]')

#solarpos = location.get_solarposition(times=pd.date_range(start=start, end=end, freq='h'))


temp_cell = pvlib.temperature.faiman(FARMS_ghi, air_temp, wind)

results_dc = pvlib.pvsystem.pvwatts_dc(FARMS_ghi,
                                      temp_cell, pdc0, 
                                      gamma_pdc = gamma_pdc, temp_ref=25)

results_dc.plot(figsize=(16,9))
plt.title('DC Power')
plt.show()

results_ac = pvlib.inverter.pvwatts(pdc=results_dc, pdc0=500,
                                    eta_inv_nom=0.961, eta_inv_ref=0.9637)
results_ac.plot(figsize=(16,9))
plt.title('AC Power')
plt.show()