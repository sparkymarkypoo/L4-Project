
def calc_spectral_modifier(material):

    import pvlib
    import pandas as pd
    import numpy as np
    import re
    
    # Import spectrum data
    spectrum_data = pd.read_csv(r'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Spectral/2021-204887-fixed_tilt.csv', skiprows=2) # Try to open this
    spectrum = spectrum_data.iloc[:,32:2034:1]/1000  # 1073
    
    # Wavelengths
    wvl_str = spectrum.columns
    wvl = 1000*np.array([float(re.sub("[^0-9.\-]","",x)) for x in wvl_str])
    wvl = wvl.round(1)
    spectrum.columns = wvl
    
    
    # Get POA
    poa = np.trapz(spectrum, x=wvl)
    poa = pd.Series(data=poa)
    
    # Get AM1.5
    am15 = pvlib.spectrum.get_am15g(wavelength=None)

    
    # Get spectral response
    spec_resp_full = pd.read_csv(rf'C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Spectral/Spectral_Response/{material}.csv', index_col=0, header=None)
            
    from scipy.interpolate import make_interp_spline
    spl = make_interp_spline(spec_resp_full.index, spec_resp_full, k=3)  # type: BSpline
    spec_resp_interp = spl(wvl)
    spec_resp_interp[spec_resp_interp < 0] = 0
    s_r = pd.Series(data=np.squeeze(spec_resp_interp), index=wvl)
        
    spec_mod = pvlib.spectrum.calc_spectral_mismatch_field(sr=s_r, e_sun=spectrum, e_ref=am15)
    spec_mod = spec_mod.fillna(0)

    #np.trapz((am15*wvl*s_r)/1239.8, x=wvl)
    return spec_mod, poa
