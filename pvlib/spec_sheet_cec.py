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

# Define parameters
celltype = 'monoSi'
pdc0 = 400
v_mp = 44.1
i_mp = 9.08
v_oc = 53.4
i_sc = 9.60
alpha_sc = 0.0005 * i_sc
beta_voc = -0.0029 * v_oc
gamma_pdc = -0.37  
cells_in_series = 6*27

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

# Calculate cell parameters
temp_cell = pvlib.temperature.faiman(data['GHI'], data['Temperature'], data['Wind Speed'])

I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = pvlib.ivtools.sdm.fit_cec_sam(
    celltype=celltype,
    v_mp = v_mp,
    i_mp = i_mp,
    v_oc = v_oc,
    i_sc = i_sc,
    alpha_sc = alpha_sc,
    beta_voc = beta_voc,
    gamma_pmp = gamma_pdc,
    cells_in_series = cells_in_series)

cec_params = pvlib.pvsystem.calcparams_cec(data['GHI'], temp_cell, alpha_sc, a_ref,
                              I_L_ref, I_o_ref, R_sh_ref, R_s, Adjust)

# Max power point
mpp = pvlib.pvsystem.max_power_point(*cec_params, method='newton')
# mpp.plot(figsize=(16,9))
# plt.show

# Scale up
system = PVSystem(modules_per_string=5, strings_per_inverter=1)
dc_scaled = system.scale_voltage_current_power(mpp)
dc_scaled.plot(figsize=(16,9))
plt.show()

# AC power with standard inverter
cec_inverters = pvlib.pvsystem.retrieve_sam('CECinverter')
inverter = cec_inverters['ABB__PVI_3_0_OUTD_S_US__208V_']
ac_results = pvlib.inverter.sandia(
    v_dc=dc_scaled.v_mp,
    p_dc=dc_scaled.p_mp,
    inverter=inverter)
ac_results.plot(figsize=(16,9))
plt.title('AC Power')
plt.show()

# # AC power with custom inverter
# results_ac = pvlib.inverter.pvwatts(pdc=dc_scaled.p_mp, pdc0=2000,
#                                     eta_inv_nom=0.961, eta_inv_ref=0.9637)
# results_ac.plot(figsize=(16,9))
# plt.title('AC Power')
# plt.show()
