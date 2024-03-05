
import pvlib
import pandas as pd
import numpy as np


# spec sheet parameters
sheet = pd.read_csv('C:/Users/mark/Documents/L4-Project-Code/Data/spec_sheets.csv', index_col=0)#, dtype=np.float64)#dtype={'cell_type':str})

out = pd.DataFrame(columns=sheet.columns, index=['alpha_sc', 'I_L_ref', 'I_o_ref', 'R_s', 'R_sh_ref', 'a_ref', 'Adjust'])

for s in sheet:

    alpha_sc = sheet[s].loc['alpha_sc'] * sheet[s].loc['isc']/100
    
    params = pvlib.ivtools.sdm.fit_cec_sam(
        celltype = s,
        v_mp = sheet[s].loc['vmp'],
        i_mp = sheet[s].loc['imp'],
        v_oc = sheet[s].loc['voc'],
        i_sc = sheet[s].loc['isc'],
        alpha_sc = alpha_sc,
        beta_voc = sheet[s].loc['beta_voc'] * sheet[s].loc['voc']/100,
        gamma_pmp = sheet[s].loc['gamma_pmp'],
        cells_in_series = sheet[s].loc['cells_in_series'])

    out[s] = (alpha_sc,) + params



# Define solar panel parameters
SAM_URL = 'https://github.com/NREL/SAM/raw/develop/deploy/libraries/CEC%20Modules.csv'
CEC_mods = pvlib.pvsystem.retrieve_sam(path=SAM_URL)
modules = CEC_mods[['First_Solar_Inc__FS_7530A_TR1','Miasole_FLEX_03_480W','Jinko_Solar_Co__Ltd_JKM340PP_72','Jinko_Solar_Co__Ltd_JKMS350M_72']]
CEC_mods = pvlib.pvsystem.retrieve_sam(name='CECMod')
modules['Xunlight_XR36_300'] = CEC_mods['Xunlight_XR36_300']
types1 = ['CdTe','CIGS','multi-Si','mono-Si','a-Si']   # THESE ONES FOR POWER SIM
modules.loc['Technology'] = types1
types2 = ['perovskite','perovskite-si','triple']                       # THESE ONES ONLY FOR POA SIM
types = types1 + types2

