import pvlib

# Define parameters from specsheet
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

poa_global = 800 # Example - can replace with series
temp_cell = 30 # as above

cec_params = pvlib.pvsystem.calcparams_cec(poa_global, temp_cell, alpha_sc, a_ref,
                              I_L_ref, I_o_ref, R_sh_ref, R_s, Adjust)