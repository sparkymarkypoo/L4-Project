#   USED TO FIND DIFFERENT MATERIALS AVAILABLE IN MODULE DATABASES

import pvlib
import matplotlib.pyplot as plt
import numpy as np

sandia_mods = pvlib.pvsystem.retrieve_sam(name='SandiaMod')
sandia_mats = sandia_mods.T['Material']
print(sandia_mats.unique())
sandia_ = sandia_mods.T
canad = sandia_[sandia_mods.index.str.contains('Canad')]



# #['Mono-c-Si' 'Multi-c-Si' 'Thin Film' 'CdTe' 'CIGS']

# cec_mods_ = pvlib.pvsystem.retrieve_sam(name='CECMod')
# cec_mods = cec_mods_.T

# xun = cec_mods[cec_mods.index.str.contains('Xun')]



# SAM_URL = 'https://github.com/NREL/SAM/raw/develop/deploy/libraries/CEC%20Modules.csv'
# CEC_mods_ = pvlib.pvsystem.retrieve_sam(path=SAM_URL)
# CEC_mods = CEC_mods_.T

# xun = CEC_mods[[CEC_mods.index.str.contains('Jinko') & CEC_mods['Technology'].str.contains('Multi-c-Si')]]

# # Jinko_Solar_Co__Ltd_JKM350PP-72-J4     poly
# # Jinko_Solar_Co__Ltd_JKMS350M-72-J4     mono
