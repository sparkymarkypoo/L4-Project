
# pd.set_option('display.max_columns', 10)
# pd.set_option('display.width', 120)

def calc_spectral_modifier(pv_name, spectrum_data):

    import pvlib
    import pandas as pd
    import numpy as np
    import re
    
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    
    # Material
    pv_name = pv_name.replace('-','')
    types = ['monoSi', 'multiSi', 'cigs', 'cdte', 'amorphous'] #'polySi', 'cis', '
    for t in types:
        if t.lower() in pv_name.lower():
            material = t
        elif 'msi' in pv_name.lower():
            material = 'multiSi'
        elif 'xsi' in pv_name.lower():
            material = 'monoSi'
        elif 'asi' in pv_name.lower():
            material = 'amorphous'
            # create a fail case?!?
      
    # Trim spectrum data
    spectrum = spectrum_data.iloc[:,32:2034:1]/1000  # 1073  
    
    
    # Wavelengths
    wvl_str = spectrum.columns
    wvl = 1000*np.array([float(re.sub("[^0-9.\-]","",x)) for x in wvl_str])
    wvl = wvl.round(1)
    spectrum.columns = wvl 
    
    
    # Get POA
    poa = np.trapz(spectrum, x=wvl)
    poa = pd.Series(data=poa, index=spectrum.index)

    
    # Find and filter nagtive (bad) values
    neg_values = (spectrum < 0).any(1)
    neg_index = neg_values.index[neg_values].tolist()
    poa.loc[neg_index] = np.nan
    
    
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
    spec_mod = spec_mod.fillna(1)
    spec_mod.loc[neg_index] = 1

    #np.trapz((am15*wvl*s_r)/1239.8, x=wvl)
    return spec_mod, poa, material
