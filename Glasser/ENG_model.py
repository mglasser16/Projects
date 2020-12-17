"""
    This file runs and executes a youyr model, calculating the cell/device/system properties of interest as a function of time.

    The code structure at this level is meant to be very simple, and delegates most functions to lower-level modules, which can be updated with new capabilties, over time.  The code:

        1 - Reads inputs and initializes the model

        2 - Calls the residual function and integrates over the user-defined
            time span.
        3 - The simulation then returns the solution vector at the end of the
            integration, which can be processed as needed to plot or analyze any quantities of interest.
"""
#%%
# Import necessary modules:
import numpy as np
from math import exp, pi
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
 #integration function for ODE system.

#%%
# Constants
F = 96485
R = 8.3145
AN = 6.0221E23 #avogadro's number
k_B = 1.38064852E-23 #boltzmann's constant
e = F/AN #elementary charge
#%%
# Inputs:
C_rate = 0.1 # How many charges per hour?
N_0 = 0 #number of initial nucleations.
alpha =0.5 #assumed value
T = 298 # K
D_Li = 3.21  #https://www.cell.com/cms/10.1016/j.joule.2018.11.016/attachment/b036fef4-0b6b-4922-9d3d-30736fd991e6/mmc1
OCV = 4 #V
i_0 =4E60 #A/m^2
rho = 2140 #kg/m3 density Li2O2 #https://iopscience.iop.org/article/10.1149/2.0351915jes/pdf
MW =  45.881/1000 #g/mol to kg/mol
sigma = 0.54 #J / m² #surface tension of electrolyte/cluster interface.
z = 1
c_st = 0.10
c_0 = 0.09 #saturated Li2O2 according to this paper https://iopscience.iop.org/article/10.1149/2.0351915jes/pdf
C_d = 80*(100*100)/1000 #microF/cm2 to microF/m2 to F/m2 https://pubs.rsc.org/en/content/articlelanding/2020/ra/d0ra00608d#!divAbstract
gamma_G = 1.2E13*100*100 #cm^2 to m2 adatom concentration, not sure what this is but took literature value ^

#calcs
A_vol = MW/(AN*rho) #m3/molecule
r_c = 2*sigma*A_vol/(z*e*OCV) + (A_vol*3/(4*pi))**(1/3) #m
A_c = 2*pi*r_c**2
b = (18*pi*A_vol**2)**(1/3)*sigma
G_c = (z*e*OCV/b)**(-3)

Area = 2*pi*r_c**2*50 #test area approximately ten times the nucleation
initial = [OCV, r_c, Area, N_0, gamma_G]
print(initial)
time_of =np.array([0,10])

#%%

def residual1(t,SV):
    #intial calcs
    V, r, A, N, adatom = SV
    n_p = 2*sigma*A_vol/(z*e*r)
    f_place = z*e/(k_B*T)
    i_dif = z*e*c_0*D_Li/r*(1-exp(-f_place*V))
    G_g= -z*e*V*G_c+b*G_c*(2/3)
    z_n = (-1/(2*pi*k_B*T))*(-2/6)*G_c**(-4/3)
    B_g=i_0*A_vol/(z*e)
    #differential
    dr_dt =  i_dif*A_vol/(z*e)
    dN_dt = z_n*B_g*adatom*exp(-G_g/(k_B*T))
    dA_dt = -(N*dr_dt*2*pi*r + dN_dt*A_c)
    dV_dt = -(i_0 - 2*pi*r**2*i_dif/A)/(C_d +z*e*f_place*adatom*exp(f_place*(OCV)))
    dadatom_dt = adatom*exp(f_place*V)
    print(dV_dt)
    return [dV_dt, dr_dt, dA_dt, dN_dt]


solution1 = solve_ivp(residual1,time_of,initial)

voltage_change = solution1.y[0]
radius_change = solution1.y[1]
Area_change = solution1.y[2]
Nucleation_change = solution1.y[3]
final_time = solution1.t
print(final_time)
#%%
print (voltage_change)
print (Area_change)
print (radius_change)
print (final_time)
#%%
plt.figure(0)
plt.plot(final_time,radius_change)
plt.xlabel("Time (s)")
plt.ylabel("Radius (m)")

#%% validation:

Area =3.14E-9*100*100 #cm^2 to m2 area
i = 0.00318 *100*100 # A/cm2 to A/m2
C_D = 8E-5*100*100 #/cm2 o F/m2
gamma_G = 1.2E13*100*100 #cm^2 to m2 adatom concentration, not sure what this is but took literature value ^
T = 300 #K
v = 1.7E-23/100/100/100 #cm3 to m3
r_0 = 1E-9
k_eff = 0.88*100 #Ohm-1, cm-1 to 1/(Ohm m)
sigma = 1E-5*100*100 #J/cm2 to J/m2
eta = 40E-10#V
N=0
time =np.array([0,1])
r_c = 10E-7/100/100 #cm to m
A_c = 2*pi*r_c**2
b = (18*pi*v**2)**(1/3)*sigma
G_c = (z*e*eta/b)**(-3)
alpha = 0.5
c_0 = 1.2E17*100*100*100 # cm ^3 to  m3
D = 10E-5/100/100 # cm2/s to m2/s
z = 1.0


validation_int =[eta, r_0, Area, gamma_G]
print (validation_int)

def validation(t,SV):
    #intial calcs
    V, r, A, adatom = SV
    n_p = 2*sigma*v/(z*e*r)
    f_place = z*e/(k_B*T)
    i_dif = 2*pi*z*e*c_0*r*D*(1-exp(-f_place*V))/(2*pi*r**2)
    # G_g= -z*e*V*G_c+b*G_c*(2/3)
    # z_n = (-1/(2*pi*k_B*T))*(-2/6)*G_c**(-4/3)
    # B_g= i*Area/(z*e)
    #differential
    # if N < 1:
    #     dr_dt =  0
    # elif N >1 :
    dr_dt =  c_0*v*D/r*(1-exp(-f_place*V))
    # if N <1:
    #     dN_dt = z_n*B_g*adatom*exp(-G_g/(k_B*T))
    # elif N >1:
    #     dN_dt = 0
    dA_dt = -dr_dt*2*pi*r
    print(A)
    # dA_dt = -(N*dr_dt*2*pi*r + dN_dt*A_c)
    # if N< 1:
    #     dV_dt = (i)/(C_D +z*e*f_place*gamma_G*exp(f_place*(V)))
    # elif N>1:
    dV_dt = (i - 2*pi*r**2*i_dif/Area)/(C_D +z*e*f_place*gamma_G*exp(f_place*(V)))
    dadatom_dt = adatom*exp(f_place*V)
    return [dV_dt, dr_dt, dA_dt, dadatom_dt]

Validation_fin = solve_ivp(validation,time,validation_int)
voltage_change = Validation_fin.y[0]
radius_change = Validation_fin.y[1]
Area_change = Validation_fin.y[2]
Nucleation_change = Validation_fin.y[3]
final_time = Validation_fin.t

#%%
plt.figure(1)
plt.plot(final_time,voltage_change)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")

plt.figure(2)
plt.plot(final_time,radius_change)
plt.xlabel("Time (s)")
plt.ylabel("radius (m)")

# plt.figure(3)
# plt.plot(final_time,Nucleation_change)
# plt.xlabel("Time (s)")
# plt.ylabel("Nucleations (#)")
plt.figure(4)
plt.plot(final_time,Area_change)
plt.xlabel("Time (s)")
plt.ylabel("Area (m3")
#%%

r_p_an = 4e-6 #m
phi_an_0 = 0 #V
C_dl_an = 1e4 #F/m2
i_o_an = 4.0  #A/m2
n_an = -1
beta_an = 0.5
H_an = 30e-6  #m
density_graphite = 2260 #kg/m3
capacity_graphite = 350 #Ah/kg
eps_graphite = .65
dPhi_eq_an = -1.6

phi_sep_0 = 1.8  #V

r_p_ca = 0.3e-6 #m
phi_ca_0 = 4.6  #V
C_dl_ca = 1e4 #F/m2
i_o_ca = 100 #A/m2
n_ca = -1
beta_ca = 0.5
H_ca = 50e-6  #m
density_LCO = 2292  #kg/m3
capacity_LCO = 175  #Ah/kg
eps_LCO = 0.65
dPhi_eq_ca = 2.6

# How deep do we want to charge/discharge?
charge_frac = 0.9


#calculated values

# Initialize:
phi_dl_an_0 = phi_an_0 - phi_sep_0
phi_dl_ca_0 = phi_ca_0 - phi_sep_0


capacity_anode = capacity_graphite*H_an*eps_graphite*density_graphite
capacity_cathode = capacity_LCO*H_ca*eps_LCO*density_LCO
capacity_area = min(capacity_anode,capacity_cathode)


t_final = charge_frac*3600./C_rate
i_ext = C_rate*capacity_area

A_fac_an = r_p_an/3/H_an/eps_graphite
A_fac_ca = r_p_ca/3/H_ca/eps_LCO

#%%
def residual(t,SV):
    dSV_dt = np.zeros_like(SV)
    eta_p = 2*sigma*A_vol/(z*e*voltage)
    eta_an = SV[0] - dPhi_eq_an
    i_Far_an = i_o_an*(exp(-n_an*F*beta_an*eta_an/R/T)
                      - exp(n_an*F*(1-beta_an)*eta_an/R/T))
    i_dl_an = i_ext*A_fac_an - i_Far_an
    dSV_dt[0] = i_dl_an/C_dl_an


    eta_ca = SV[1] - dPhi_eq_ca
    i_Far_ca = i_o_ca*(exp(-n_ca*F*beta_ca*eta_ca/R/T)
                      - exp(n_ca*F*(1-beta_ca)*eta_ca/R/T))
    i_dl_ca = -i_ext*A_fac_ca - i_Far_ca


    dSV_dt[1] = i_dl_ca/C_dl_ca

    return dSV_dt

# Either directly in this file, or in a separate file that you import, define:
#   - A residual function called 'residual'
#   - An array 'time_span' which has [0, t_final] where t_final is the total
#       length of time you want the model to simulate.
#   - An intial solution vector SV_0
#%%
SV_0 = np.array([phi_dl_an_0, phi_dl_ca_0])

time_span = np.array([0,t_final])

solution = solve_ivp(residual,time_span,SV_0,rtol=1e-6, atol=1e-8)

for var in solution.y:
    plt.plot(solution.t,var)

plt.legend(['Anode double layer','Cathode double layer'])


# %%
