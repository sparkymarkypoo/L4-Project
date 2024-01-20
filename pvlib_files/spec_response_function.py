
def calc_spectral_modifier():

    import pvlib
    import pandas as pd
    import numpy as np

    # Wavelength values
    wvl_5 = np.arange(280,400,0.5)
    wvl_10 = np.arange(400,1201,1)
    wvl = np.concatenate([wvl_5, wvl_10])
    
    # Import spectrum data
    spectrum_data = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Grid/2021-204887-fixed_tilt.csv', skiprows=2) # Try to open this
    spectrum = spectrum_data.iloc[:,32:1073:1]/1000
    spectrum.columns = wvl
    
    # Create time
    time_pd = spectrum_data[['Year', 'Month', 'Day', 'Hour', 'Minute']]
    time = pd.to_datetime(time_pd)
    spectrum.index = time # NEED TO ADJUST FOR TZ
    
    # Get AM1.5
    am15_full = pvlib.spectrum.get_am15g(wavelength=None)
    am15 = am15_full[280.0:1200.0]
    
    # Get spectral response
    spec_resp_full = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Grid/Default Dataset.csv', index_col=0, header=None)
    from scipy.interpolate import make_interp_spline
    spl = make_interp_spline(spec_resp_full.index, spec_resp_full, k=3)  # type: BSpline
    spec_resp_interp = spl(wvl)
    spec_resp_interp[spec_resp_interp < 0] = 0
    s_r = pd.Series(data=np.squeeze(spec_resp_interp), index=wvl)
    
    spec_mod = pvlib.spectrum.calc_spectral_mismatch_field(sr=s_r, e_sun=spectrum, e_ref=am15)
    spec_mod = spec_mod.fillna(0)
    
    return spec_mod
