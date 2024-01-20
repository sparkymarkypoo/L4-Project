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
gamma_pdc = -0.0037
temp_ref = 25

# Choose time range:
year = 2022
start = np.datetime64(f'{year}-05-26T00:00')
end = np.datetime64(f'{year}-05-30T00:00')

# Get NREL (FARMS) data
latitude = 36.626
longitude = -116.018 
interval = 5 # in minutes
data = NRELdata(start, end, year, latitude, longitude, interval)
FARMS_time = np.arange(start, end+5, dtype=f'datetime64[{interval}m]')

# Location
surface_azimuth=180
surface_tilt=37

#solarpos = location.get_solarposition(times=pd.date_range(start=start, end=end, freq='h'))


temp_cell = pvlib.temperature.faiman(data['GHI'], data['Temperature'], data['Wind Speed'])

results_dc = pvlib.pvsystem.pvwatts_dc(data['GHI'],
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