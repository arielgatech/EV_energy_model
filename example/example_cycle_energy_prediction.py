# -*- coding: utf-8 -*-
"""
Created on Mon Mar 04 13:28:37 2019

@author: xxu312
"""

import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import time
from os import listdir

### constant value, no need to change ###
gravity_acceleration = 9.81  # m/s2
air_density = 1.1985 # kg/m3
rolling_resist_coeff_1 = 0.008 # constant
rolling_resist_coeff_2 = 0.00012  # per m/s

### define EV attributes ###


# Nissan Leaf
EV_100mile_attr = {"type": "BEV_100mile",
                    "vehicle_weight": 1659, #kg
                   "drag_coeff":0.315,
                   "frontal_area":2.755, #m^2
                   "battery_size_kwh":30.4128,
                   "max_soc":1,
                   "min_soc":0}

# Tesla model s
EV_300mile_attr = {"type": "BEV_300mile",
                    "vehicle_weight": 2270, #kg
                   "drag_coeff":0.3,
                   "frontal_area":2.832, #m^2
                   "battery_size_kwh":101.177,                   
                   "max_soc":1,
                   "min_soc":0} # per m/s

# Toyota mirai
FCEV_attr = {"type": "FCEV",
                    "vehicle_weight": 1760, #kg
                   "drag_coeff":0.3,
                   "frontal_area":2.786, #m^2
                   "battery_size_kwh":1.823472,
                   "max_soc":0.7,
                   "min_soc":0.4} # per m/s

# Toyota prius prime
PS_PHEV_attr = {"type": "PS_PHEV",
                    "vehicle_weight": 1712, #kg
                   "drag_coeff":0.311,
                   "frontal_area":2.372, #m^2
                   "battery_size_kwh":8.1147744,
                   "max_soc":0.9,
                   "min_soc":0.1} # per m/s

# Toyota prius
PS_HEV_attr = {"type": "PS_HEV",
                    "vehicle_weight": 1669, #kg
                   "drag_coeff":0.311,
                   "frontal_area":2.372, #m^2
                   "battery_size_kwh":1.2636,
                   "max_soc":0.9,
                   "min_soc":0.1} # per m/s

# Chevy volt
SER_PHEV_attr = {"type": "SER_PHEV",
                    "vehicle_weight": 1893, #kg
                   "drag_coeff":0.3,
                   "frontal_area":2.565, #m^2
                   "battery_size_kwh":14.888772,
                   "max_soc":0.9,
                   "min_soc":0.1} # per m/s

# Ford fusion hybrid
PAR_HEV_attr = {"type": "PAR_HEV",
                    "vehicle_weight": 1639.7, #kg
                   "drag_coeff":0.3,
                   "frontal_area":2.25, #m^2
                   "battery_size_kwh":1.458,
                   "max_soc":0.9,
                   "min_soc":0.1} # per m


### Trained parameter from BN model ###
BEV100_param = [1.9466, 930.2749, 1.3168, 765.0286]
BEV300_param = [2.6535, 687.9322, 1.7740, 601.6535]

PS_PHEV_eng_on_param = [-2.67677801, 4.24002192e-02, 9.15611588e-05, 1.65119099e-04, -1.93716152]
PS_PHEV_mot1_pos_param = [-0.11713582, 0.00028122, 0.04196995]

PS_HEV_eng_on_param = [-1.77820524,  3.14728592e-02,  4.22270081e-04, -0.22471421]
PS_HEV_mot1_pos_param = [-0.13633639,  0.00031071, 0.32515293]

SER_PHEV_eng_on_param = [-3.06319594, 8.08387735e-02, 3.75583185e-05, 1.66369333e-05, -2.27318062]
SER_PHEV_mot1_pos_param = [0.0, 0.00024091, 0.04576871]

PAR_HEV_eng_on_param = [-2.67418185,  1.54580705,  1.81889166,  9.46418049e-05, 0.69051686]
PAR_HEV_mot1_pos_param = [0.699, 0.371, 0.097, 0.018, 0.925, 0.038, 0.668, 0.992]


## define EV class for store EV attributes ##
class electric_vehicle:
    def __init__(self,vid,veh):
        self.id = vid
        self.type = veh['type']
        self.weight = veh['vehicle_weight']
        self.drag_coeff = veh['drag_coeff']
        self.frontal_area = veh['frontal_area']
        self.battery_size = veh['battery_size_kwh']
        self.max_soc = veh['max_soc']
        self.min_soc = veh['min_soc']
                
## basic calculation##
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def hev_soc_mapping(in_soc):
#    out_soc = in_soc - 0.2
    out_soc = (in_soc <= 0.5) * (0.5 * in_soc - 0.1) + (in_soc > 0.5) * (1.5 * in_soc - 0.45)
    return out_soc


def update_soc(initial_soc, energy_consumption, battery_size):
    energy_available = initial_soc * battery_size
    energy_remaining = energy_available - energy_consumption
    latest_soc = energy_remaining / battery_size
    return latest_soc

## power demand calculation ##
def vehicle_power_calculator(speed, acc, grade, weight, frontal_area, drag_coeff):
    rolling_resistance_coeff = rolling_resist_coeff_1 + speed * rolling_resist_coeff_2
    rolling_resistance = rolling_resistance_coeff * weight * gravity_acceleration * np.cos(grade)
    aerodynamic_resistance = 0.5 * air_density * frontal_area * drag_coeff * (speed ** 2)
    uphill_drag = weight * gravity_acceleration * np.sin(grade)
    acceleration_load = weight * acc
    vehicle_power_demand_watt = (acceleration_load + aerodynamic_resistance + rolling_resistance + uphill_drag) * speed
    return vehicle_power_demand_watt

## BEV energy use calculation ##
def ALLEV_energy_calculator_pd(cycle, elec_coeff, hvac_step_size):
    cycle['VSP_pos'] = 1 * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)
    cycle['VSP_pos_value'] = cycle['VSP_est'] * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)
    cycle['VSP_neg'] = 1 * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    cycle['VSP_neg_value'] = cycle['VSP_est'] * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    x_elec = cycle[['VSP_pos_value', 'VSP_pos', 'VSP_neg_value', 'VSP_neg']]
    cycle['elec_est'] = x_elec.dot(elec_coeff)
    if hvac_step_size > 0:     
        cycle = cycle.reset_index()
        cycle['VSP_w'] = 1000 * np.round(cycle['VSP_est']/1000, 0)     
        plus_elec = pd.merge(cycle, elec_hvac_factor, on = ['VSP_w'], how = 'left')
        cycle['elec_est'] += plus_elec['delta_elec_energy_rate'] 
    return cycle

## FCEV energy use calculation ##
def FCEV_energy_calculator_pd(cycle, time_step, hvac_step_size):
    cycle['VSP_pos'] = 1 * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)
    cycle['VSP_pos_value'] = cycle['VSP_est'] * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)
    cycle['VSP_neg'] = 1 * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    cycle['VSP_neg_value'] = cycle['VSP_est'] * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    cycle['SOC_vsp_pos'] = cycle['soc_simulation'] * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)
    cycle['SOC_vsp_neg'] = cycle['soc_simulation'] * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    x_fuel = cycle[['VSP_neg', 'SOC_vsp_pos', 'VSP_pos_value', 'VSP_pos']]
    x_fuel_coeff = [3892.0, -1.28898e+05, 4.0256, 97903.299]
    cycle['fuel_est'] = x_fuel.dot(x_fuel_coeff)
    if hvac_step_size > 0:
        cycle = cycle.reset_index()
        cycle['VSP_w'] = 1000 * np.round(cycle['VSP_est']/1000, 0)        
        plus_fuel = pd.merge(cycle, fuel_hvac_factor, on = ['VSP_w'], how = 'left')
        cycle['fuel_est'] += plus_fuel['delta_fuel_energy_rate']          
    cycle['fuel_vsp_neg'] = 0.1 * cycle['fuel_est'] * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)    
    cycle['fuel_vsp_pos'] = 0.1 * cycle['fuel_est'] * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)    
    x_elec = cycle[['SOC_vsp_neg', 'VSP_neg_value', 'fuel_vsp_neg', 'VSP_neg', 'SOC_vsp_pos', 'VSP_pos_value', 'fuel_vsp_pos', 'VSP_pos']]
    x_elec_coeff = [3.8071e+03, 0.79744, -3.0205, -3029.3013, 1.07364e+04, 1.4756, -3.7569, -4848.2654]
    cycle['elec_est'] = x_elec.dot(x_elec_coeff)
    if hvac_step_size > 0:     
        plus_elec = pd.merge(cycle, elec_hvac_factor, on = ['VSP_w'], how = 'left')
        cycle['elec_est'] += plus_elec['delta_elec_energy_rate']   
    return cycle


## Parallel hybrid energy use calculation ##
def PAR_HEV_energy_calculator_pd(cycle, target_soc, eng_on_param, mot1_pos_param, hvac_step_size):
    cycle['high_SOC'] = 1 * (cycle['soc_simulation'] >= target_soc + 0.05) + 0 * (cycle['soc_simulation'] < target_soc + 0.05)
    cycle['medium_SOC'] = 1 * (cycle['soc_simulation'] < target_soc + 0.05) * (cycle['soc_simulation'] >= target_soc) + 0 * (cycle['soc_simulation'] < target_soc)
    + 0 * (cycle['soc_simulation'] >= target_soc + 0.05)
    cycle['low_SOC'] = 1 * (cycle['soc_simulation'] <= target_soc) + 0 * (cycle['soc_simulation'] > target_soc)
    cycle['constant'] = 1
    x_eng_on = cycle[['high_SOC', 'medium_SOC', 'low_SOC', 'VSP_est', 'constant']]
    cycle['eng_on_score'] = x_eng_on.dot(eng_on_param)
    cycle['eng_on_prob'] = sigmoid(cycle['eng_on_score'])
    cycle['bin_1'] = (cycle['soc_simulation'] < 0.45) * (cycle['Speed(mph)'] < 3.7) * (cycle['VSP_est'] < 0.0)
    cycle['bin_2'] = (cycle['soc_simulation'] < 0.45) * (cycle['Speed(mph)'] < 3.7) * (cycle['VSP_est'] >= 0.0)
    cycle['bin_3'] = (cycle['soc_simulation'] < 0.45) * (cycle['Speed(mph)'] >= 3.7) * (cycle['VSP_est'] < 230.0)
    cycle['bin_4'] = (cycle['soc_simulation'] < 0.45) * (cycle['Speed(mph)'] >= 3.7) * (cycle['VSP_est'] >= 230.0)
    cycle['bin_5'] = (cycle['soc_simulation'] >= 0.45) * (cycle['Speed(mph)'] < 3.7) * (cycle['VSP_est'] <= -100.0)
    cycle['bin_6'] = (cycle['soc_simulation'] >= 0.45) * (cycle['Speed(mph)'] >= 3.7) * (cycle['VSP_est'] <= -100.0)
    cycle['bin_7'] = (cycle['soc_simulation'] >= 0.45) * (cycle['VSP_est'] < 0.0) * (cycle['VSP_est'] >= -100.0)
    cycle['bin_8'] = (cycle['soc_simulation'] >= 0.45) * (cycle['VSP_est'] >= 0.0)
    x_mot1_pos = cycle[['bin_1', 'bin_2', 'bin_3', 'bin_4', 'bin_5', 'bin_6', 'bin_7', 'bin_8']]
    cycle['mot1_pos_score'] = x_mot1_pos.dot(mot1_pos_param)
    cycle['mot1_pos_prob'] = sigmoid(cycle['mot1_pos_score'])
    cycle['mot1_pos_eng_on_prob'] = cycle['eng_on_prob'] * cycle['mot1_pos_prob']
    cycle['mot1_neg_eng_on_prob'] = cycle['eng_on_prob'] * (1 - cycle['mot1_pos_prob'])   
    # engine off energy
    cycle['fuel_eng_off'] = 0.0
    cycle['VSP_pos'] = 1 * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)
    cycle['VSP_pos_value'] = cycle['VSP_est'] * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)
    cycle['VSP_neg'] = 1 * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    cycle['VSP_neg_value'] = cycle['VSP_est'] * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    x_elec_eng_off = cycle[['VSP_pos_value', 'VSP_pos', 'VSP_neg_value', 'VSP_neg']]
    x_elec_eng_off_coeff = [1.962524, 570.9734794284032, 1.2682302, 550.8562089109209]
    cycle['elec_eng_off'] = x_elec_eng_off.dot(x_elec_eng_off_coeff)
    # engine on energy
    x_fuel_eng_on_mot1_pos = cycle[['soc_simulation', 'VSP_est', 'constant']]
    x_fuel_eng_on_mot1_pos_coeff = [-4.20632612e+04, 3.97007485, 25795.593398371613]
    cycle['fuel_eng_on_mot1_pos'] = x_fuel_eng_on_mot1_pos.dot(x_fuel_eng_on_mot1_pos_coeff)
    cycle['x1_elec_mot1_pos'] = cycle['soc_simulation'] * (cycle['soc_simulation'] >= target_soc) + 0 * (cycle['soc_simulation'] < target_soc)
    cycle['x2_elec_mot1_pos'] = cycle['VSP_est'] * (cycle['soc_simulation'] >= target_soc) + 0 * (cycle['soc_simulation'] < target_soc)
    cycle['x3_elec_mot1_pos'] = 1 * (cycle['soc_simulation'] >= target_soc) + 0 * (cycle['soc_simulation'] < target_soc)
    cycle['x4_elec_mot1_pos'] = cycle['VSP_est'] * (cycle['soc_simulation'] < target_soc) + 0 * (cycle['soc_simulation'] >= target_soc)
    x_elec_eng_on_mot1_pos = cycle[['x1_elec_mot1_pos', 'x2_elec_mot1_pos', 'x3_elec_mot1_pos', 'x4_elec_mot1_pos', 'low_SOC']]
    x_elec_eng_on_mot1_pos_coeff = [2.97321643e+04, 2.32793218e-01, -12204.642108379012, 0.22400676, 484.2424644001493]
    cycle['elec_eng_on_mot1_pos'] = x_elec_eng_on_mot1_pos.dot(x_elec_eng_on_mot1_pos_coeff)
    cycle['x1_fuel_mot1_neg'] = cycle['soc_simulation'] * cycle['VSP_pos']
    x_fuel_eng_on_mot1_neg = cycle[['VSP_neg', 'x1_fuel_mot1_neg', 'VSP_pos_value', 'VSP_pos']]
    x_fuel_eng_on_mot1_neg_coeff = [805.2133096730364, -2.78938755e+05, 4.36362293, 140439.00530550053]
    cycle['fuel_eng_on_mot1_neg'] = x_fuel_eng_on_mot1_neg.dot(x_fuel_eng_on_mot1_neg_coeff)
    cycle['x1_elec_mot1_neg'] = cycle['soc_simulation'] * cycle['VSP_neg']
    x_elec_eng_on_mot1_neg = cycle[['x1_elec_mot1_neg', 'VSP_neg_value', 'VSP_neg', 'x1_fuel_mot1_neg', 'VSP_pos']]
    x_elec_eng_on_mot1_neg_coeff = [2.32002319e+03, 1.32649525, -385.50192025424167, 8.56073873e+04,  -38795.65]
    cycle['elec_eng_on_mot1_neg'] = x_elec_eng_on_mot1_neg.dot(x_elec_eng_on_mot1_neg_coeff)
    # HVAC adjustment
    if hvac_step_size > 0:
        cycle = cycle.reset_index()
        cycle['VSP_w'] = 1000 * np.round(cycle['VSP_est']/1000, 0)        
        plus_fuel_under_motoring = pd.merge(cycle, fuel_hvac_factor_motoring, on = ['VSP_w'], how = 'left')
        cycle['fuel_eng_on_mot1_pos'] += plus_fuel_under_motoring['delta_fuel_energy_rate']        
        plus_fuel_under_generating = pd.merge(cycle, fuel_hvac_factor_generating, on = ['VSP_w'], how = 'left')
        cycle['fuel_eng_on_mot1_neg'] += plus_fuel_under_generating['delta_fuel_energy_rate']

        plus_elec_under_eng_off = pd.merge(cycle, elec_hvac_factor_eng_off, on = ['VSP_w'], how = 'left')
        cycle['elec_eng_off'] += plus_elec_under_eng_off['delta_elec_energy_rate']
        plus_elec_under_eng_on_mot_pos = pd.merge(cycle, elec_hvac_factor_motoring, on = ['VSP_w'], how = 'left')
        cycle['elec_eng_on_mot1_pos'] += plus_elec_under_eng_on_mot_pos['delta_elec_energy_rate']
        plus_elec_under_eng_on_mot_neg = pd.merge(cycle, elec_hvac_factor_generating, on = ['VSP_w'], how = 'left')
        cycle['elec_eng_on_mot1_neg'] += plus_elec_under_eng_on_mot_neg['delta_elec_energy_rate']
    # sum energy use    
    cycle['fuel_est'] = cycle['fuel_eng_off'] * (1 - cycle['eng_on_prob']) + \
    cycle['fuel_eng_on_mot1_pos'] * cycle['mot1_pos_eng_on_prob'] + cycle['fuel_eng_on_mot1_neg'] * cycle['mot1_neg_eng_on_prob']
    cycle['fuel_est'] = cycle['fuel_est'] * (cycle['fuel_est'] >= 0) + 0 * (cycle['fuel_est'] < 0)
    cycle['elec_est'] = cycle['elec_eng_off'] * (1 - cycle['eng_on_prob']) + \
    cycle['elec_eng_on_mot1_pos'] * cycle['mot1_pos_eng_on_prob'] + cycle['elec_eng_on_mot1_neg'] * cycle['mot1_neg_eng_on_prob']
    return cycle


## Series hybrid energy use calculation ##
def SER_PHEV_energy_calculator_pd(cycle, target_soc, eng_on_param, mot1_pos_param, hvac_step_size):
    cycle['high_SOC'] = 1 * (cycle['soc_simulation'] >= target_soc) + 0 * (cycle['soc_simulation'] < target_soc)
    cycle['constant'] = 1
    x_eng_on = cycle[['high_SOC', 'Speed(mph)', 'max_VSP', 'min_VSP', 'constant']]
    cycle['eng_on_score'] = x_eng_on.dot(eng_on_param)
    cycle['eng_on_prob'] = sigmoid(cycle['eng_on_score'])
    x_mot1_pos = cycle[['Speed(mph)', 'VSP_est', 'constant']]
    cycle['mot1_pos_score'] = x_mot1_pos.dot(mot1_pos_param)
    cycle['mot1_pos_prob'] = sigmoid(cycle['mot1_pos_score'])
    cycle['mot1_pos_eng_on_prob'] = cycle['eng_on_prob'] * cycle['mot1_pos_prob']
    cycle['mot1_neg_eng_on_prob'] = cycle['eng_on_prob'] * (1 - cycle['mot1_pos_prob'])
    # engine off energy
    cycle['fuel_eng_off'] = 0.0
    cycle['VSP_pos'] = 1 * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)
    cycle['VSP_pos_value'] = cycle['VSP_est'] * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)
    cycle['VSP_neg'] = 1 * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    cycle['VSP_neg_value'] = cycle['VSP_est'] * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    x_elec_eng_off = cycle[['VSP_pos_value', 'VSP_pos', 'VSP_neg_value', 'VSP_neg']]
    x_elec_eng_off_coeff = [2.28165196, 476.1935775980219, 1.56366896, 538.2556373432371]
    cycle['elec_eng_off'] = x_elec_eng_off.dot(x_elec_eng_off_coeff)
    # engine on energy
    cycle['upper_SOC'] = 1 * (cycle['soc_simulation'] >= 0.36) + 0 * (cycle['soc_simulation'] < 0.36)
    cycle['botton_SOC'] = cycle['soc_simulation'] * (cycle['soc_simulation'] < 0.36) + 0 * (cycle['soc_simulation'] >= 0.36)
    x_eng_on_mot1_pos = cycle[['upper_SOC', 'botton_SOC', 'VSP_est', 'constant']]
    x_fuel_eng_on_mot1_pos_coeff = [-2.74509128e+05, -9.09039559e+05,  4.75220283, 288932.8393509499]
    x_elec_eng_on_mot1_pos_coeff = [7.80310767e+04, 2.57519913e+05, 1.39387618e-01, -78908.847443407]
    cycle['fuel_eng_on_mot1_pos'] = x_eng_on_mot1_pos.dot(x_fuel_eng_on_mot1_pos_coeff)
    cycle['elec_eng_on_mot1_pos'] = x_eng_on_mot1_pos.dot(x_elec_eng_on_mot1_pos_coeff)
    cycle['low_SOC'] = cycle['soc_simulation'] * (cycle['soc_simulation'] < 0.3) + 0 * (cycle['soc_simulation'] >= 0.3)

    x_eng_on_mot1_neg =  cycle[['low_SOC', 'high_SOC',  'VSP_est', 'constant']]
    x_fuel_eng_on_mot1_neg_coeff = [-1.18233539e+06, -3.54980210e+05,  3.90625854, 385522.6214090834]
    x_elec_eng_on_mot1_neg_coeff = [ 2.25179949e+05,  6.75313253e+04, -1.79147887e-02, -75129.8799274783]
    cycle['fuel_eng_on_mot1_neg'] = x_eng_on_mot1_neg.dot(x_fuel_eng_on_mot1_neg_coeff)
    cycle['elec_eng_on_mot1_neg'] = x_eng_on_mot1_neg.dot(x_elec_eng_on_mot1_neg_coeff)
    # HVAC adjustment
    if hvac_step_size > 0:
        cycle = cycle.reset_index()
        cycle['VSP_w'] = 1000 * np.round(cycle['VSP_est']/1000, 0)        
        plus_fuel_under_motoring = pd.merge(cycle, fuel_hvac_factor_motoring, on = ['VSP_w'], how = 'left')
        cycle['fuel_eng_on_mot1_pos'] += plus_fuel_under_motoring['delta_fuel_energy_rate']        
        plus_fuel_under_generating = pd.merge(cycle, fuel_hvac_factor_generating, on = ['VSP_w'], how = 'left')
        cycle['fuel_eng_on_mot1_neg'] += plus_fuel_under_generating['delta_fuel_energy_rate']

        plus_elec_under_eng_off = pd.merge(cycle, elec_hvac_factor_eng_off, on = ['VSP_w'], how = 'left')
        cycle['elec_eng_off'] += plus_elec_under_eng_off['delta_elec_energy_rate']
        plus_elec_under_eng_on_mot_pos = pd.merge(cycle, elec_hvac_factor_motoring, on = ['VSP_w'], how = 'left')
        cycle['elec_eng_on_mot1_pos'] += plus_elec_under_eng_on_mot_pos['delta_elec_energy_rate']
        plus_elec_under_eng_on_mot_neg = pd.merge(cycle, elec_hvac_factor_generating, on = ['VSP_w'], how = 'left')
        cycle['elec_eng_on_mot1_neg'] += plus_elec_under_eng_on_mot_neg['delta_elec_energy_rate']
    # sum energy use    
    cycle['fuel_est'] = cycle['fuel_eng_off'] * (1 - cycle['eng_on_prob']) + \
    cycle['fuel_eng_on_mot1_pos'] * cycle['mot1_pos_eng_on_prob'] + cycle['fuel_eng_on_mot1_neg'] * cycle['mot1_neg_eng_on_prob']
    cycle['fuel_est'] = cycle['fuel_est'] * (cycle['fuel_est'] >= 0) + 0 * (cycle['fuel_est'] < 0)
    cycle['elec_est'] = cycle['elec_eng_off'] * (1 - cycle['eng_on_prob']) + \
    cycle['elec_eng_on_mot1_pos'] * cycle['mot1_pos_eng_on_prob'] + cycle['elec_eng_on_mot1_neg'] * cycle['mot1_neg_eng_on_prob']
    return cycle


## Power-split plug-in hybrid energy use calculation ##
def PS_PHEV_energy_calculator_pd(cycle, target_soc, eng_on_param, mot1_pos_param, hvac_step_size):
    # veh_control_prob
    cycle['high_SOC'] = 1 * (cycle['soc_simulation'] >= target_soc) + 0 * (cycle['soc_simulation'] < target_soc)
    cycle['constant'] = 1
    x_eng_on = cycle[['high_SOC', 'Speed(mph)', 'max_VSP', 'min_VSP', 'constant']]
    cycle['eng_on_score'] = x_eng_on.dot(eng_on_param)
    cycle['eng_on_prob'] = sigmoid(cycle['eng_on_score'])
    x_mot1_pos = cycle[['Speed(mph)', 'VSP_est', 'constant']]
    cycle['mot1_pos_score'] = x_mot1_pos.dot(mot1_pos_param)
    cycle['mot1_pos_prob'] = sigmoid(cycle['mot1_pos_score'])
    cycle['mot1_pos_eng_on_prob'] = cycle['eng_on_prob'] * cycle['mot1_pos_prob']
    cycle['mot1_neg_eng_on_prob'] = cycle['eng_on_prob'] * (1 - cycle['mot1_pos_prob'])
    # engine off energy
    cycle['fuel_eng_off'] = 0.0
    cycle['VSP_pos'] = 1 * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)
    cycle['VSP_pos_value'] = cycle['VSP_est'] * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)
    cycle['VSP_neg'] = 1 * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    cycle['VSP_neg_value'] = cycle['VSP_est'] * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    x_elec_eng_off = cycle[['VSP_pos_value', 'VSP_pos', 'VSP_neg_value', 'VSP_neg']]
    x_elec_eng_off_coeff = [2.45157107, 144.38601445994937, 1.33505183, 346.41790648524875]
    cycle['elec_eng_off'] = x_elec_eng_off.dot(x_elec_eng_off_coeff)
    # engine on / motoring energy
    cycle['upper_SOC'] = 1 * (cycle['soc_simulation'] >= 0.36) + 0 * (cycle['soc_simulation'] < 0.36)
    cycle['botton_SOC'] = cycle['soc_simulation'] * (cycle['soc_simulation'] < 0.36) + 0 * (cycle['soc_simulation'] >= 0.36)
    x_eng_on_mot1_pos = cycle[['upper_SOC', 'botton_SOC', 'VSP_est', 'constant']]
    x_fuel_eng_on_mot1_pos_coeff = [-3.38597796e+05, -1.16994098e+06, 3.3584077, 363839.0015780216]
    cycle['fuel_eng_on_mot1_pos'] = x_eng_on_mot1_pos.dot(x_fuel_eng_on_mot1_pos_coeff)
    x_elec_eng_on_mot1_pos_coeff = [1.69653646e+05, 5.79011468e+05, 5.57667998e-01, -173651.62228421587]
    cycle['elec_eng_on_mot1_pos'] = x_eng_on_mot1_pos.dot(x_elec_eng_on_mot1_pos_coeff)   
    # engine on / generating energy
    cycle['low_SOC'] = cycle['soc_simulation'] * (cycle['soc_simulation'] < 0.3) + 0 * (cycle['soc_simulation'] >= 0.3)
    cycle['medium_SOC'] = cycle['soc_simulation'] * (cycle['soc_simulation'] < 0.36) * (cycle['soc_simulation'] >= 0.3) + 0 * (cycle['soc_simulation'] < 0.3) + + 0 * (cycle['soc_simulation'] >= 0.36)
    x_fuel_eng_on_mot1_neg = cycle[['low_SOC', 'medium_SOC', 'upper_SOC', 'VSP_est', 'constant']]    
    x_fuel_eng_on_mot1_neg_coeff = [-2.03315356e+05, -2.76856773e+05, -7.16715346e+04, 4.14674728, 87600.84724097612]
    cycle['fuel_eng_on_mot1_neg'] = x_fuel_eng_on_mot1_neg.dot(x_fuel_eng_on_mot1_neg_coeff)
    x_elec_eng_on_mot1_neg = cycle[['Speed(mph)', 'low_SOC', 'medium_SOC', 'upper_SOC', 'VSP_est', 'constant']]    
    x_elec_eng_on_mot1_neg_coeff = [3.47681620e+02, -6.92437852e+04, -3.36729340e+04, -1.46818108e+04, -3.51383515e-01, 3199.503220777504]
    cycle['elec_eng_on_mot1_neg'] = x_elec_eng_on_mot1_neg.dot(x_elec_eng_on_mot1_neg_coeff)
    # HVAC adjustment
    if hvac_step_size > 0:
        cycle = cycle.reset_index()
        cycle['VSP_w'] = 1000 * np.round(cycle['VSP_est']/1000, 0)        
        plus_fuel_under_motoring = pd.merge(cycle, fuel_hvac_factor_motoring, on = ['VSP_w'], how = 'left')
        cycle['fuel_eng_on_mot1_pos'] += plus_fuel_under_motoring['delta_fuel_energy_rate']        
        plus_fuel_under_generating = pd.merge(cycle, fuel_hvac_factor_generating, on = ['VSP_w'], how = 'left')
        cycle['fuel_eng_on_mot1_neg'] += plus_fuel_under_generating['delta_fuel_energy_rate']

        plus_elec_under_eng_off = pd.merge(cycle, elec_hvac_factor_eng_off, on = ['VSP_w'], how = 'left')
        cycle['elec_eng_off'] += plus_elec_under_eng_off['delta_elec_energy_rate']
        plus_elec_under_eng_on_mot_pos = pd.merge(cycle, elec_hvac_factor_motoring, on = ['VSP_w'], how = 'left')
        cycle['elec_eng_on_mot1_pos'] += plus_elec_under_eng_on_mot_pos['delta_elec_energy_rate']
        plus_elec_under_eng_on_mot_neg = pd.merge(cycle, elec_hvac_factor_generating, on = ['VSP_w'], how = 'left')
        cycle['elec_eng_on_mot1_neg'] += plus_elec_under_eng_on_mot_neg['delta_elec_energy_rate']

    # sum energy use    
    cycle['fuel_est'] = cycle['fuel_eng_off'] * (1 - cycle['eng_on_prob']) + \
    cycle['fuel_eng_on_mot1_pos'] * cycle['mot1_pos_eng_on_prob'] + cycle['fuel_eng_on_mot1_neg'] * cycle['mot1_neg_eng_on_prob']
    cycle['fuel_est'] = cycle['fuel_est'] * (cycle['fuel_est'] >= 0) + 0 * (cycle['fuel_est'] < 0)
    cycle['elec_est'] = cycle['elec_eng_off'] * (1 - cycle['eng_on_prob']) + \
    cycle['elec_eng_on_mot1_pos'] * cycle['mot1_pos_eng_on_prob'] + cycle['elec_eng_on_mot1_neg'] * cycle['mot1_neg_eng_on_prob']
    return cycle

## Power-split hybrid energy use calculation ##
def PS_HEV_energy_calculator_pd(cycle, target_soc, eng_on_param, mot1_pos_param, hvac_step_size):
    # veh_control_prob
    cycle['high_SOC'] = 1 * (cycle['soc_simulation'] >= target_soc) + 0 * (cycle['soc_simulation'] < target_soc)
    cycle['low_SOC'] = 0 * (cycle['soc_simulation'] >= target_soc) + cycle['soc_simulation'] * (cycle['soc_simulation'] < target_soc)
    cycle['constant'] = 1
    x_eng_on = cycle[['high_SOC', 'Speed(mph)', 'VSP_est', 'constant']]
    cycle['eng_on_score'] = x_eng_on.dot(eng_on_param)
    cycle['eng_on_prob'] = sigmoid(cycle['eng_on_score'])
    x_mot1_pos = cycle[['Speed(mph)', 'VSP_est', 'constant']]
    cycle['mot1_pos_score'] = x_mot1_pos.dot(mot1_pos_param)
    cycle['mot1_pos_prob'] = sigmoid(cycle['mot1_pos_score'])
    cycle['mot1_pos_eng_on_prob'] = cycle['eng_on_prob'] * cycle['mot1_pos_prob']
    cycle['mot1_neg_eng_on_prob'] = cycle['eng_on_prob'] * (1 - cycle['mot1_pos_prob'])
    # engine off energy
    cycle['fuel_eng_off'] = 0.0
    cycle['VSP_pos'] = 1 * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)
    cycle['VSP_pos_value'] = cycle['VSP_est'] * (cycle['VSP_est'] >= 0) + 0 * (cycle['VSP_est'] < 0)
    cycle['VSP_neg'] = 1 * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    cycle['VSP_neg_value'] = cycle['VSP_est'] * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    cycle['high_soc_VSP_neg_value'] = cycle['high_SOC'] * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    cycle['low_soc_VSP_neg_value'] = cycle['low_SOC'] * (cycle['VSP_est'] < 0) + 0 * (cycle['VSP_est'] >= 0)
    x_elec_eng_off = cycle[['VSP_pos_value', 'VSP_pos', 'high_soc_VSP_neg_value', 'low_soc_VSP_neg_value', 'VSP_neg_value', 'VSP_neg']]
    x_elec_eng_off_coeff = [2.54922345, 609.6860982699941, -1.03505851e+04, -1.85091790e+04,  1.28339855, 11660.532292454865]
    cycle['elec_eng_off'] = x_elec_eng_off.dot(x_elec_eng_off_coeff)
    # engine on / motoring energy
    cycle['upper_SOC'] = 1 * (cycle['soc_simulation'] >= 0.6) + 0 * (cycle['soc_simulation'] < 0.6)
    cycle['botton_SOC'] = cycle['soc_simulation'] * (cycle['soc_simulation'] < 0.4) + 0 * (cycle['soc_simulation'] >= 0.4)
    cycle['medium_SOC'] = cycle['soc_simulation'] * (cycle['soc_simulation'] < 0.6) * (cycle['soc_simulation'] >= 0.4) + 0 * (cycle['soc_simulation'] < 0.4) + + 0 * (cycle['soc_simulation'] >= 0.6)
    x_fuel_eng_on_mot1_pos = cycle[['high_SOC', 'low_SOC', 'VSP_est', 'constant']]
    x_fuel_eng_on_mot1_pos_coeff = [-4.73614873e+04, -9.01088995e+04,  4.31096633, 50467.52777484011]
    cycle['fuel_eng_on_mot1_pos'] = x_fuel_eng_on_mot1_pos.dot(x_fuel_eng_on_mot1_pos_coeff)
    x_elec_eng_on_mot1_pos = cycle[['botton_SOC', 'medium_SOC', 'upper_SOC', 'Acceleration(mph/s)', 'constant']]    
    x_elec_eng_on_mot1_pos_coeff = [30636.29317032, 29277.90727329, 17821.16717848,  1062.08837062, -14086.10000612037]
    cycle['elec_eng_on_mot1_pos'] = x_elec_eng_on_mot1_pos.dot(x_elec_eng_on_mot1_pos_coeff)   
    # engine on / generating energy
    x_fuel_eng_on_mot1_neg = cycle[['botton_SOC', 'medium_SOC', 'upper_SOC', 'VSP_est', 'constant']]    
    x_fuel_eng_on_mot1_neg_coeff = [-7.19037068e+04, -7.64297827e+04, -4.38215248e+04,  4.00333903, 48986.66871479346]
    cycle['fuel_eng_on_mot1_neg'] = x_fuel_eng_on_mot1_neg.dot(x_fuel_eng_on_mot1_neg_coeff)
    x_elec_eng_on_mot1_neg = cycle[['Speed(mph)', 'botton_SOC', 'medium_SOC', 'upper_SOC', 'constant']]    
    x_elec_eng_on_mot1_neg_coeff = [105.36625859, 22780.20637935, 24752.42083166, 15557.4291121, -14753.878598382245]
    cycle['elec_eng_on_mot1_neg'] = x_elec_eng_on_mot1_neg.dot(x_elec_eng_on_mot1_neg_coeff)
    # HVAC adjustment
    if hvac_step_size > 0:
        cycle = cycle.reset_index()
        cycle['VSP_w'] = 1000 * np.round(cycle['VSP_est']/1000, 0)        
        plus_fuel_under_motoring = pd.merge(cycle, fuel_hvac_factor_motoring, on = ['VSP_w'], how = 'left')
        cycle['fuel_eng_on_mot1_pos'] += plus_fuel_under_motoring['delta_fuel_energy_rate']        
        plus_fuel_under_generating = pd.merge(cycle, fuel_hvac_factor_generating, on = ['VSP_w'], how = 'left')
        cycle['fuel_eng_on_mot1_neg'] += plus_fuel_under_generating['delta_fuel_energy_rate']

        plus_elec_under_eng_off = pd.merge(cycle, elec_hvac_factor_eng_off, on = ['VSP_w'], how = 'left')
        cycle['elec_eng_off'] += plus_elec_under_eng_off['delta_elec_energy_rate']
        plus_elec_under_eng_on_mot_pos = pd.merge(cycle, elec_hvac_factor_motoring, on = ['VSP_w'], how = 'left')
        cycle['elec_eng_on_mot1_pos'] += plus_elec_under_eng_on_mot_pos['delta_elec_energy_rate']
        plus_elec_under_eng_on_mot_neg = pd.merge(cycle, elec_hvac_factor_generating, on = ['VSP_w'], how = 'left')
        cycle['elec_eng_on_mot1_neg'] += plus_elec_under_eng_on_mot_neg['delta_elec_energy_rate']

    # sum energy use    
    cycle['fuel_est'] = cycle['fuel_eng_off'] * (1 - cycle['eng_on_prob']) + \
    cycle['fuel_eng_on_mot1_pos'] * cycle['mot1_pos_eng_on_prob'] + cycle['fuel_eng_on_mot1_neg'] * cycle['mot1_neg_eng_on_prob']
    cycle['fuel_est'] = cycle['fuel_est'] * (cycle['fuel_est'] >= 0) + 0 * (cycle['fuel_est'] < 0)
    cycle['elec_est'] = cycle['elec_eng_off'] * (1 - cycle['eng_on_prob']) + \
    cycle['elec_eng_on_mot1_pos'] * cycle['mot1_pos_eng_on_prob'] + cycle['elec_eng_on_mot1_neg'] * cycle['mot1_neg_eng_on_prob']
    return cycle

# <codecell>
### start of main function ###

def main():
    start_time = time.time()
    path = "TESTING_CYCLE"  # path to input driving cycle
    hvac_load = 3 # define HVAC load in kw
    veh_type = "SER_PHEV"# define vehicle type, 'BEV100' -> 100-mile nissan leaf, 'BEV300' -> 300-mile tesla, 
    #PAR_HEV -> Ford fusion,  SER_PHEV -> Chevy volt, PS_PHEV -> Prius prime, PS_HEV -> Prius
    out_path = veh_type + '_AC/' # define output directory
    all_cycle = listdir(path)
    hvac_step_size = np.round((hvac_load - 0.5)/0.5, 0)
    vid = 0
    for cyc in all_cycle: # loop through each cycle
        if cyc == ".DS_Store":
            continue
        else:
            input_file = cyc
            vid += 1
            print("get cycle energy " + cyc)
            
        ### get HVAC adjustment factors ###    
        global fuel_hvac_factor, elec_hvac_factor
        if hvac_step_size > 0:
            if veh_type in ['PAR_HEV', 'SER_PHEV', 'PS_PHEV', 'PS_HEV']:
                global fuel_hvac_factor_motoring, fuel_hvac_factor_generating, elec_hvac_factor_eng_off, elec_hvac_factor_motoring, elec_hvac_factor_generating
                fuel_hvac_factor_motoring, fuel_hvac_factor_generating, elec_hvac_factor_eng_off, elec_hvac_factor_motoring, elec_hvac_factor_generating = get_hev_hvac_adjust(hvac_step_size, veh_type)
            elif veh_type == 'FCEV':
                fuel_hvac_factor, elec_hvac_factor = get_fcev_hvac_adjust(hvac_step_size, veh_type)
            elif veh_type in ['BEV100', 'BEV300']:
                elec_hvac_factor = get_bev_hvac_adjust(hvac_step_size, veh_type)
        
        ### compute energy use ###
        global sample_cycle_out
        if veh_type in ['PAR_HEV', 'SER_PHEV', 'PS_PHEV', 'PS_HEV']:
            for soc in range(17): # loop through SOC between [10%, 90%], increment 5%
                initial_soc = 0.1 + 0.05 * soc
                soc_level = int(5 * soc + 10)
                print(str(soc_level))
                road_grade = 0.0 # assuming flat terrain
                sample_cycle_out = get_hev_energy(path, input_file, vid, hvac_step_size, veh_type, initial_soc, road_grade) 
                sample_cycle_out.to_csv(out_path + str(soc_level) + '_' + input_file, sep = ',', index = False)
        elif veh_type == 'FCEV':
            for soc in range(7):
                initial_soc = 0.4 + 0.05 * soc # loop through SOC between [40%, 70%], increment 5%
                soc_level = int(5*soc+40)
                print(str(soc_level))
                road_grade = 0.0
                sample_cycle_out = get_fcev_energy(path, input_file, vid, hvac_step_size, veh_type, initial_soc, road_grade) 
                sample_cycle_out.to_csv(out_path + str(soc_level) + '_' + input_file, sep = ',', index = False)
        elif veh_type in ['BEV100', 'BEV300']:
            initial_soc = 0.9
            road_grade = 0.0
            sample_cycle_out = get_bev_energy(path, input_file, vid, hvac_step_size, veh_type, initial_soc, road_grade) 
            sample_cycle_out.to_csv(out_path + input_file, sep = ',', index = False)
        print("total time is "  + str(time.time() - start_time) )     
        
def get_hev_hvac_adjust(hvac_step_size, veh_type):    
    fuel_hvac_factor = read_csv('HVAC_results/' + veh_type + '_fuel_hvac_factor.csv', sep = ',')
    fuel_hvac_factor 
    fuel_hvac_factor['delta_fuel_energy_rate'] *= hvac_step_size
    fuel_hvac_factor_motoring = fuel_hvac_factor.loc[fuel_hvac_factor['mot_control'] == 1]
    fuel_hvac_factor_generating = fuel_hvac_factor.loc[fuel_hvac_factor['mot_control'] == 0]
    
    elec_hvac_factor = read_csv('HVAC_results/' + veh_type + '_elec_hvac_factor.csv', sep = ',')
    elec_hvac_factor['delta_elec_energy_rate'] *= hvac_step_size
    elec_hvac_factor_eng_off = elec_hvac_factor.loc[elec_hvac_factor['eng_on'] == 0]
    elec_hvac_factor_motoring = elec_hvac_factor.loc[(elec_hvac_factor['eng_on'] == 1) & (elec_hvac_factor['mot_control'] == 1)]
    elec_hvac_factor_generating = elec_hvac_factor.loc[(elec_hvac_factor['eng_on'] == 1) & (elec_hvac_factor['mot_control'] == 0)]
    return fuel_hvac_factor_motoring, fuel_hvac_factor_generating, elec_hvac_factor_eng_off, elec_hvac_factor_motoring, elec_hvac_factor_generating

def get_fcev_hvac_adjust(hvac_step_size, veh_type):    
    fuel_hvac_factor = read_csv('HVAC_results/' + veh_type + '_fuel_hvac_factor.csv', sep = ',')
    fuel_hvac_factor['delta_fuel_energy_rate'] *= hvac_step_size    
    elec_hvac_factor = read_csv('HVAC_results/' + veh_type + '_elec_hvac_factor.csv', sep = ',')
    elec_hvac_factor['delta_elec_energy_rate'] *= hvac_step_size
    return fuel_hvac_factor, elec_hvac_factor

def get_bev_hvac_adjust(hvac_step_size, veh_type):    
    elec_hvac_factor = read_csv('HVAC_results/' + veh_type + '_elec_hvac_factor.csv', sep = ',')
    elec_hvac_factor['delta_elec_energy_rate'] *= hvac_step_size
    return elec_hvac_factor

def get_hev_energy(path, input_file, vid, hvac_step_size, veh_type, initial_soc, road_grade, time_resolusion = 1):    
    update_period = int(60 / time_resolusion) # update SOC every 600 datapoints 
    EV = {}    
    global sample_cycle
    sample_cycle = read_csv(path + '/' + input_file, sep = ',')
    sample_cycle['Speed(mph)'] = sample_cycle['speed(m/s)'] * 2.23694
    sample_cycle.loc[:, 'Acceleration(m/s2)'] = sample_cycle['speed(m/s)'] - sample_cycle['speed(m/s)'].shift(1)
    sample_cycle.loc[0, 'Acceleration(m/s2)'] = 0
    sample_cycle.loc[:, 'Acceleration(mph/s)'] = sample_cycle.loc[:, 'Acceleration(m/s2)'] * 2.23694
    sample_cycle['road_grade(rad)'] = np.arctan(road_grade / 100.0)    
    if veh_type == 'PS_PHEV': 
        EV[vid] = electric_vehicle(vid, PS_PHEV_attr)
    elif veh_type == 'PS_HEV': 
        EV[vid] = electric_vehicle(vid, PS_HEV_attr)        
    elif veh_type == 'SER_PHEV':
        EV[vid] = electric_vehicle(vid, SER_PHEV_attr)   
    elif veh_type == 'PAR_HEV':
        EV[vid] = electric_vehicle(vid, PAR_HEV_attr)   
        
    sample_cycle.loc[:, 'power_est'] = sample_cycle.apply(lambda row: vehicle_power_calculator(row['speed(m/s)'], row['Acceleration(m/s2)'], row['road_grade(rad)'], EV[vid].weight, EV[vid].frontal_area, EV[vid].drag_coeff), axis=1)
    sample_cycle.loc[:, 'VSP_est'] = sample_cycle.loc[:, 'power_est'] / EV[vid].weight * 1000.0
    rolling_window = int(3 * (time_resolusion == 1) + 2 / time_resolusion * (time_resolusion < 1))
    rolling_top = rolling_window - 1
#    print(rolling_window, rolling_top)
    sample_cycle['max_VSP'] = sample_cycle['VSP_est'].rolling(window=rolling_window).max()
    sample_cycle['max_VSP'].fillna(float(sample_cycle.iloc[rolling_top]['max_VSP']), inplace = True)
    sample_cycle['min_VSP'] = sample_cycle['VSP_est'].rolling(window=rolling_window).min()
    sample_cycle['min_VSP'].fillna(float(sample_cycle.iloc[rolling_top]['min_VSP']), inplace = True)
#    initial_soc = float(sample_cycle.head(1)['SOC'])
    sample_cycle['soc_simulation'] = initial_soc
    sample_cycle['fuel_est'] = None
    sample_cycle['elec_est'] = None
    elec_consumption_period = 0
    num_of_chunks = len(sample_cycle)//update_period + 1
    sample_cycle_split = np.array_split(sample_cycle, num_of_chunks)
    sample_cycle_out = None
    new_soc = initial_soc
    global cycle_to_process_new
    for chunk in range(num_of_chunks):
#        print(chunk)
        cycle_to_process = sample_cycle_split[chunk]
        new_soc = EV[vid].max_soc * (new_soc > EV[vid].max_soc) + new_soc * (new_soc >= EV[vid].min_soc) * (new_soc <= EV[vid].max_soc) + EV[vid].min_soc * (new_soc < EV[vid].min_soc)
        cycle_to_process.loc[:, 'soc_simulation'] = new_soc
        if EV[vid].type == 'PS_HEV':
            target_soc = 0.5   
            cycle_to_process_new = PS_HEV_energy_calculator_pd(cycle_to_process, target_soc, PS_HEV_eng_on_param, PS_HEV_mot1_pos_param, hvac_step_size) 
        if EV[vid].type == 'PS_PHEV':
            target_soc = 0.3   
            cycle_to_process_new = PS_PHEV_energy_calculator_pd(cycle_to_process, target_soc, PS_PHEV_eng_on_param, PS_PHEV_mot1_pos_param, hvac_step_size)    
        elif EV[vid].type == 'SER_PHEV':
            target_soc = 0.3   
            cycle_to_process_new = SER_PHEV_energy_calculator_pd(cycle_to_process, target_soc, SER_PHEV_eng_on_param, SER_PHEV_mot1_pos_param, hvac_step_size)    
        elif EV[vid].type == 'PAR_HEV':
            target_soc = 0.45   
            cycle_to_process_new = PAR_HEV_energy_calculator_pd(cycle_to_process, target_soc, PAR_HEV_eng_on_param, PAR_HEV_mot1_pos_param, hvac_step_size)    

        elec_consumption_period = cycle_to_process_new.loc[:, 'elec_est'].sum() * time_resolusion / 3600 / 1000
        new_soc = update_soc(new_soc, elec_consumption_period, EV[vid].battery_size)
        if sample_cycle_out is None:
            sample_cycle_out = cycle_to_process_new
        else:
            sample_cycle_out = pd.concat([sample_cycle_out, cycle_to_process_new], sort=False)
    sample_cycle_out.loc[:, 'fuel_est'] *= time_resolusion
    sample_cycle_out.loc[:, 'elec_est'] *= time_resolusion   
    sample_cycle_out = sample_cycle_out[['speed(m/s)', 'Acceleration(m/s2)', 'Speed(mph)', 'road_grade(rad)', 'VSP_est', 'soc_simulation', 'fuel_est', 'elec_est']]
    return sample_cycle_out

def get_fcev_energy(path, input_file, vid, hvac_step_size, veh_type, initial_soc, road_grade, time_resolusion = 1.0):    
    update_period = int(60 / time_resolusion) # update SOC every 60 datapoints 
    EV = {}    
    global sample_cycle
    sample_cycle = read_csv(path + '/' + input_file, sep = ',')
    sample_cycle['Speed(mph)'] = sample_cycle['speed(m/s)'] * 2.23694
    sample_cycle.loc[:, 'Acceleration(m/s2)'] = sample_cycle['speed(m/s)'] - sample_cycle['speed(m/s)'].shift(1)
    sample_cycle.loc[0, 'Acceleration(m/s2)'] = 0
    sample_cycle['road_grade(rad)'] = np.arctan(road_grade / 100.0)
    EV[vid] = electric_vehicle(vid, FCEV_attr)        
    sample_cycle.loc[:, 'power_est'] = sample_cycle.apply(lambda row: vehicle_power_calculator(row['speed(m/s)'], row['Acceleration(m/s2)'], row['road_grade(rad)'], EV[vid].weight, EV[vid].frontal_area, EV[vid].drag_coeff), axis=1)
    sample_cycle.loc[:, 'VSP_est'] = sample_cycle.loc[:, 'power_est'] / EV[vid].weight * 1000.0
    rolling_window = int(3 * (time_resolusion == 1) + 2 / time_resolusion * (time_resolusion < 1))
    rolling_top = rolling_window - 1
    sample_cycle['max_VSP'] = sample_cycle['VSP_est'].rolling(window=rolling_window).max()
    sample_cycle['max_VSP'].fillna(float(sample_cycle.iloc[rolling_top]['max_VSP']), inplace = True)
    sample_cycle['min_VSP'] = sample_cycle['VSP_est'].rolling(window=rolling_window).min()
    sample_cycle['min_VSP'].fillna(float(sample_cycle.iloc[rolling_top]['min_VSP']), inplace = True)
#    initial_soc = float(sample_cycle.head(1)['SOC'])
    sample_cycle['soc_simulation'] = initial_soc
    sample_cycle['fuel_est'] = None
    sample_cycle['elec_est'] = None
    elec_consumption_period = 0
    num_of_chunks = len(sample_cycle)//update_period + 1
    sample_cycle_split = np.array_split(sample_cycle, num_of_chunks)
    sample_cycle_out = None
    new_soc = initial_soc
    global cycle_to_process_new
    for chunk in range(num_of_chunks):
#        print(chunk)
        cycle_to_process = sample_cycle_split[chunk]
        cycle_to_process.loc[:, 'soc_simulation'] = 0.9 * (new_soc > 0.9) + new_soc * (new_soc >= 0.1) * (new_soc <= 0.9) + 0.1 * (new_soc < 0.1)
        cycle_to_process_new = FCEV_energy_calculator_pd(cycle_to_process, time_resolusion, hvac_step_size)     
        elec_consumption_period = cycle_to_process_new.loc[:, 'elec_est'].sum() * time_resolusion / 3600.0 / 1000.0
        new_soc = update_soc(new_soc, elec_consumption_period, EV[vid].battery_size)
        if sample_cycle_out is None:
            sample_cycle_out = cycle_to_process_new
        else:
            sample_cycle_out = pd.concat([sample_cycle_out, cycle_to_process_new], sort=False)
    sample_cycle_out.loc[:, 'fuel_est'] *= time_resolusion
    sample_cycle_out.loc[:, 'elec_est'] *= time_resolusion   
    sample_cycle_out = sample_cycle_out[['speed(m/s)', 'Acceleration(m/s2)', 'Speed(mph)', 'road_grade(rad)', 'VSP_est', 'soc_simulation', 'fuel_est', 'elec_est']]
    return sample_cycle_out

def get_bev_energy(path, input_file, vid, hvac_step_size, veh_type, initial_soc, road_grade, time_resolusion = 1):    
    
    EV = {}    
    global sample_cycle
    sample_cycle = read_csv(path + '/' + input_file, sep = ',')
    sample_cycle['Speed(mph)'] = sample_cycle['speed(m/s)'] * 2.23694
    sample_cycle.loc[:, 'Acceleration(m/s2)'] = sample_cycle['speed(m/s)'] - sample_cycle['speed(m/s)'].shift(1)
    sample_cycle.loc[0, 'Acceleration(m/s2)'] = 0
    sample_cycle['road_grade(rad)'] = np.arctan(road_grade / 100.0)
    if veh_type == 'BEV100':
        EV[vid] = electric_vehicle(vid, EV_100mile_attr)
    elif veh_type == 'BEV300':
        EV[vid] = electric_vehicle(vid, EV_300mile_attr)        
    sample_cycle.loc[:, 'power_est'] = sample_cycle.apply(lambda row: vehicle_power_calculator(row['speed(m/s)'], row['Acceleration(m/s2)'], row['road_grade(rad)'], EV[vid].weight, EV[vid].frontal_area, EV[vid].drag_coeff), axis=1)
    sample_cycle.loc[:, 'VSP_est'] = sample_cycle.loc[:, 'power_est'] / EV[vid].weight * 1000.0
    sample_cycle['elec_est'] = None
    global cycle_to_process_new
    if veh_type == 'BEV100':
        cycle_to_process_new = ALLEV_energy_calculator_pd(sample_cycle, BEV100_param, hvac_step_size) 
    elif veh_type == 'BEV300':
        cycle_to_process_new = ALLEV_energy_calculator_pd(sample_cycle, BEV300_param, hvac_step_size) 
    cycle_to_process_new.loc[:, 'elec_est'] *= time_resolusion
#    initial_soc = float(sample_cycle.head(1)['SOC'])
    cycle_to_process_new.loc[:, 'elec_est_cum'] = cycle_to_process_new.loc[:, 'elec_est'].cumsum()
    cycle_to_process_new.loc[:, 'elec_est_cum_kwh'] = cycle_to_process_new.loc[:, 'elec_est_cum'] / 1000 / 3600 
    cycle_to_process_new['soc_simulation'] = (EV[vid].battery_size * initial_soc - cycle_to_process_new['elec_est_cum_kwh']) / EV[vid].battery_size

    return cycle_to_process_new
    
main()