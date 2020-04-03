# -*- coding: utf-8 -*-
"""
Created on Mon Mar 04 11:54:06 2019

@author: xxu312
"""

from pandas import read_csv
import pandas as pd

#BEV_elec_adjust_files = ['BEV100_elec_HVAC_adjust.csv', 'BEV300_elec_HVAC_adjust.csv']

#FCEV_elec_adjust_files = ['FCEV_elec_HVAC_adjust.csv']

#HEV_elec_adjust_files = ['EREV_elec_HVAC_adjust.csv', 'PAR_HEV_elec_HVAC_adjust.csv', 'PS_PHEV_elec_HVAC_adjust.csv', 'PS_HEV_elec_HVAC_adjust.csv']

#HEV_fuel_adjust_files = ['FCEV_fuel_HVAC_adjust.csv', 'EREV_fuel_HVAC_adjust.csv', 'PAR_HEV_fuel_HVAC_adjust.csv', 'PS_PHEV_fuel_HVAC_adjust.csv', 'PS_HEV_fuel_HVAC_adjust.csv']

HEV_elec_adjust_files = ['PS_HEV_elec_HVAC_adjust.csv']

HEV_fuel_adjust_files = ['PS_HEV_fuel_HVAC_adjust.csv']

vsp_lookup = read_csv('VSP_lookup.csv', sep = ',')
delta_hvac_power = 500.0

#for BEV_file in BEV_elec_adjust_files:
#    vehicle_type = BEV_file.split("_")[0]
#    BEV_adjust_factor = read_csv(BEV_file, sep = ',')
#    BEV_adjust_factor['delta_elec_energy_rate'] = None
#    BEV_adjust_factor.loc[BEV_adjust_factor['t_value'].abs() <= 1.96, 'HVAC_adjust_rate'] = 0.0
#    BEV_adjust_factor.loc[:, 'delta_elec_energy_rate'] = BEV_adjust_factor.loc[:, 'HVAC_adjust_rate'] * delta_hvac_power
#    BEV_adjust_factor = pd.merge(BEV_adjust_factor, vsp_lookup, on = ['VSP_group'], how = 'outer') 
#    BEV_adjust_factor_out = BEV_adjust_factor[['VSP_kw', 'HVAC_adjust_rate', 't_value', 'delta_elec_energy_rate']]
#    BEV_adjust_factor_out.to_csv(vehicle_type + '_elec_hvac_factor.csv', sep = ',')
    
    
for HEV_file in HEV_elec_adjust_files:
    vehicle_type = HEV_file.split("_")[0]
    BEV_adjust_factor = read_csv(HEV_file, sep = ',')
    BEV_adjust_factor['delta_elec_energy_rate'] = None
    BEV_adjust_factor.loc[BEV_adjust_factor['t_value'].abs() <= 1.96, 'HVAC_adjust_rate'] = 0.0
    BEV_adjust_factor.loc[:, 'delta_elec_energy_rate'] = BEV_adjust_factor.loc[:, 'HVAC_adjust_rate'] * delta_hvac_power
    BEV_adjust_factor = pd.merge(BEV_adjust_factor, vsp_lookup, on = ['VSP_group'], how = 'outer') 
    BEV_adjust_factor_out = BEV_adjust_factor[['VSP_kw', 'eng_on', 'mot_control', 'HVAC_adjust_rate', 't_value', 'delta_elec_energy_rate']]
    BEV_adjust_factor_out.to_csv(vehicle_type + '_elec_hvac_factor.csv', sep = ',') 
#    
#for FCEV_file in FCEV_elec_adjust_files:
#    vehicle_type = FCEV_file.split("_")[0]
#    BEV_adjust_factor = read_csv(FCEV_file, sep = ',')
#    BEV_adjust_factor['delta_elec_energy_rate'] = None
#    BEV_adjust_factor.loc[BEV_adjust_factor['t_value'].abs() <= 1.96, 'HVAC_adjust_rate'] = 0.0
#    BEV_adjust_factor.loc[:, 'delta_elec_energy_rate'] = BEV_adjust_factor.loc[:, 'HVAC_adjust_rate'] * delta_hvac_power
#    BEV_adjust_factor = pd.merge(BEV_adjust_factor, vsp_lookup, on = ['VSP_group'], how = 'outer') 
#    BEV_adjust_factor_out = BEV_adjust_factor[['VSP_kw', 'fc_on', 'mot_control', 'HVAC_adjust_rate', 't_value', 'delta_elec_energy_rate']]
#    BEV_adjust_factor_out.to_csv(vehicle_type + '_elec_hvac_factor.csv', sep = ',')    
#    
for HEV_file in HEV_fuel_adjust_files:
    vehicle_type = HEV_file.split("_")[0]
    BEV_adjust_factor = read_csv(HEV_file, sep = ',')
    BEV_adjust_factor['delta_fuel_energy_rate'] = None
    BEV_adjust_factor.loc[BEV_adjust_factor['t_value'].abs() <= 1.96, 'HVAC_adjust_rate'] = 0.0
    BEV_adjust_factor.loc[:, 'delta_fuel_energy_rate'] = BEV_adjust_factor.loc[:, 'HVAC_adjust_rate'] * delta_hvac_power
    BEV_adjust_factor = pd.merge(BEV_adjust_factor, vsp_lookup, on = ['VSP_group'], how = 'outer') 
    BEV_adjust_factor_out = BEV_adjust_factor[['VSP_kw', 'mot_control', 'HVAC_adjust_rate', 't_value', 'delta_fuel_energy_rate']]
    BEV_adjust_factor_out.to_csv(vehicle_type + '_fuel_hvac_factor.csv', sep = ',')      
#    break