#%%
# Useful libraries
import datetime as dt
import shutil
import math as m
import numpy as np
import pandas as pd
import random as ran
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp

"""=========================== User input variables ==========================="""
time = 200 #s  - input desired duration of simulation
i_0 = 0.0004 #A/m2 - galvanostatic current, based on experimental conditions?
phi1 = 0.9 #V -taken from Hw 6 code, I know it's not quite realistic.
phi2 = 0.6
# Initial conditions
n_0 = 10  #input how many nucleii are already present
c_k = 1.2 #concentrations of solute - should be in kmol/m³
T = 298 #K
V_elect = 0.0005  #volume electrolyte m³
alpha = 0.5
beta = (1-alpha)
F = 96485
R = 8.3145
AN = 6.0221E23 #avogadro's number
k_B = 1.38064852E-23 #boltzmann's constant
e = F/AN #elementary charge
H_an = 2E-3 # m  based on experimental
H_sep = 25E-3 #m
H_cath = 2E-3 #m
C_elyte = 1 #kmol/m3


# Thermo properties
c_k_sat = 3 # kmol/m3 Solute concentration at 100% saturation - should be in kmol/m³ (this is a guess)
surf_energy = 0.54 #J / m2 surface energy of solid product, should be Li₂O₂, temporary data for lithium, ... (http://crystalium.materialsvirtuallab.org/) can probably be done with cantera?
MW = 45.881 #kg/kmol
den =2310 #kg/m3 2.31 #g/cm3
D_li = 7.3e-10 #m2/s https://www.cell.com/cms/10.1016/j.joule.2018.11.016/attachment/b036fef4-0b6b-4922-9d3d-30736fd991e6/mmc1
ConLi_el = 1.085 #S/m https://iopscience.iop.org/article/10.1149/2.0351915jes/pdf
Con_el = 0.014*100 #S/cm to S/m https://www.cell.com/cms/10.1016/j.joule.2018.11.016/attachment/b036fef4-0b6b-4922-9d3d-30736fd991e6/mmc1

# Growth reaction parameters
k_grow = 2    # Rate coefficient (mol/m²/s)
n = 1           # Reaction order (-)
k_r = 1   # Rate coefficient

# Constants
R = 8314.4 #J/K / kmol
"""=================== Initial calculations, problem set-up ==================="""

# Initial parameter calculations:
i_bv = i_0*(exp(alpha*F*(phi1-phi2)/(R*T))-exp(-beta*F*(phi1-phi2)/(R*T)))
S = c_k/c_k_sat
mol_vol = den/MW #kmol/m³
Con_li_gen = i_bv*e/z/H_sep

#intializing the solution vector

initial = {}
initial['r_0'] = 2**surf_energy*(mol_vol)/(R*T*m.log(S)) #m
#initial['n_0'] =0 # nucleii
initial['S'] = S #unitless
initial['con_elyte'] = c_k
sol_vec = list(initial.values())  # solution vector
print(sol_vec)
#Thermo Adjustments
k_rev = k_r*m.exp(2*surf_energy*mol_vol/(R*T*initial['r_0']))

A_spec = (10*initial['r_0'])**2/(V_elect)
#Reaction surface area/volume of electrolyte, used if the rate of reactions is mol/m², I think used in nucleation, Specific surface of reaction (m²/m³) using r_0 for scale

int_volume =  2/3*initial['r_0']**3
#initiual volume of a nucleation

t_n = 0.15 #https://www.cell.com/cms/10.1016/j.joule.2018.11.016/attachment/b036fef4-0b6b-4922-9d3d-30736fd991e6/mmc1
v_plus_Li =1

#%%



"""======================== Define the residual function ========================"""
def residual(t, solution):
    r, s, con_ely = solution #indicates variable array because I forget
    dr_dt = MW/den*(k_grow*(s)**n-k_rev*(2*m.pi*r**2))
    ds_dt = i_bv*t_n/(z*v_plus_Li*F) - n_0*(dr_dt * 2 * m.pi * r**2)*mol_vol/V_elect/c_k_sat + D_Li*(con_ely -s*c_k_sat)/c_k_sat # distribute concentration change into total electrolyte
    dcon_elyte = Con_li_gen - i_bv*t_n/(z*v_plus_Li*F) - D_Li*(con_ely -s*c_k_sat)
#    drad_dt = (.5*m.tanh(180*(conc-1)+.5))*mol_vol*k_grow*(conc-1)**n
#    dconc_dt = - (.5*m.tanh(180*(conc-1)+.5))*n_0*(drad_dt*2*m.pi*radius**2)/mol_vol/V_elect/co_k
#    drad_dt = m.tanh(30*(conc-1))*mol_vol*k_grow*(conc-1)**n
#    dconc_dt = - m.tanh(30*(conc-1))*n_0*(drad_dt*2*m.pi*radius**2)/mol_vol/V_elect/co_k  distrute change in mass to electrolyte
    return [dr_dt, ds_dt, dcon_elyte]

"""========================== Run the simulation =========================="""
solution = solve_ivp(residual, [0, time], sol_vec) #growth senario

"""============================ Post-processing ============================"""


radius = solution.y[0]
concentrations = solution.y[1]
print(concentrations)
t = solution.t

#%%

r_range = max(radius) + min(radius)
max_rad = max(radius)
x = [ran.random()*r_range for i in range(n_0)]
y = [ran.random()*r_range for i in range(n_0)]

#%%
with PdfPages('output' +  dt.datetime.now().strftime("%Y%m%d") + '.pdf') as pdf:
    plt.figure(0)
    plt.plot(t,radius)
    plt.xlabel("Time (s)")
    plt.ylabel("Radius (m)")
    pdf.savefig()
    plt.show()
    plt.close()

    plt.figure(1)
    plt.plot(t,concentrations)
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration (C/Ck)")
    pdf.savefig()
    plt.show()
    plt.close()

    for i in range(0,len(t), int(0.01*max(t))):
        plt.figure(i+2)
        plt.scatter(x, y, s=np.ones_like(x)*3000*radius[i])
        plt.axis([0.0, max_rad, 0.0, max_rad])
        pdf.savefig()
        plt.close()
# %%

shutil.copy(__file__, __file__+ dt.datetime.now().strftime("%Y%m%d")+".txt")

# %%
