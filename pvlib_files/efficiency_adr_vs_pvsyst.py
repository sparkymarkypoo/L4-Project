import numpy as np
import matplotlib.pyplot as plt

from pvlib.pvsystem import calcparams_pvsyst, max_power_point
from pvlib.pvarray import fit_pvefficiency_adr, pvefficiency_adr

from timeit import timeit

pvsyst_params = {'alpha_sc': 0.0015,
                 'gamma_ref': 1.20585,
                 'mu_gamma': -9.41066e-05,
                 'I_L_ref': 5.9301,
                 'I_o_ref': 2.9691e-10,
                 'R_sh_ref': 1144,
                 'R_sh_0': 3850,
                 'R_s': 0.6,
                 'cells_in_series': 96,
                 'R_sh_exp': 5.5,
                 'EgRef': 1.12,
                 }

G_REF = 1000
T_REF = 25

params_stc = calcparams_pvsyst(G_REF, T_REF, **pvsyst_params)
mpp_stc = max_power_point(*params_stc)

P_REF = mpp_stc['p_mp']

g, t = np.meshgrid(np.linspace(100, 1100, 11),
                   np.linspace(0, 75, 4))

adjusted_params = calcparams_pvsyst(g, t, **pvsyst_params)
mpp = max_power_point(*adjusted_params)
p_mp = mpp['p_mp']

eta_rel_pvs = (p_mp / P_REF) / (g / G_REF)


# Calc ADR efficiency and errors
adr_params = fit_pvefficiency_adr(g, t, eta_rel_pvs, dict_output=True)
for k, v in adr_params.items():
        eta_rel_adr = pvefficiency_adr(g, t, **adr_params)
mbe = np.mean(eta_rel_adr - eta_rel_pvs)
rmse = np.sqrt(np.mean(np.square(eta_rel_adr - eta_rel_pvs)))


# PLOT
plt.figure()
plt.plot(g.flat, eta_rel_pvs.flat, 'oc', ms=8)
plt.plot(g.flat, eta_rel_adr.flat, '.k')
plt.grid(alpha=0.5)
plt.xlim(0, 1200)
plt.ylim(0.7, 1.1)

plt.xlabel('Irradiance [W/m²]')
plt.ylabel('Relative efficiency [-]')
plt.legend(['PVsyst model output', 'ADR model fit'], loc='lower right')
plt.title('Differences: mean %.5f, RMS %.5f' % (mbe, rmse))
plt.show()