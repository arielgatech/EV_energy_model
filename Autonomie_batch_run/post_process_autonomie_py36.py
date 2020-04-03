# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 09:28:54 2018

@author: arielxxd
"""
import numpy as np
from os import listdir
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

#final_cycle_in = None
#final_cycle_out = None
veh_type = 'BEV200'
all_output = listdir(veh_type + '_hvac')
cycle_processed = []
for afile in all_output:
    if afile == '.DS_Store':
        continue
    print("processing autonomie output " + afile)
    input_dir = veh_type + '_hvac/' + afile +  '/' + 'data.mat'
    autonomie_out = loadmat(input_dir)
    all_keys = autonomie_out.keys()
    cycle_name = autonomie_out['ans']
    # copied from Matlab
    soc_min = 0.1
    soc_max = 0.9
    soc_target = 0.1
    battery_cap = 48.733 * 342 * 3.6 / 1000
    
#    powertrain = str(autonomie_out['pwt'].tolist()[0])
    vehicle_weight_kg = autonomie_out['veh'][0,0][0][0][0][0][0][0][0][0][0][0][0][0]
#    print vehicle_weight_kg
#    print autonomie_out['veh'].flatten('C')
#    if cycle_name in cycle_processed:
#        continue
#    else:
#    cycle_processed.append(cycle_name)
    time_step = autonomie_out['time_simu']
    time_step_out = pd.DataFrame(time_step, columns = ['time(s)'])
    SOC = autonomie_out['ess_plant_soc_simu']
    SOC_out = pd.DataFrame(SOC, columns = ['SOC'])
    initial_soc = int(100 * SOC[0])
##    if SOC[0]>0.5:
##        plt.plot(SOC)
    tractive_power = autonomie_out['drv_ctrl_pwr_dmd_simu']
    tractive_power_out = pd.DataFrame(tractive_power, columns = ['tracpower(watt)'])
    tractive_power_out['VSP'] = tractive_power_out['tracpower(watt)'] / (vehicle_weight_kg * 0.001)
#    fc_fuel = autonomie_out['fc_plant_fuel_rate_simu']
#    fc_fuel_out = pd.DataFrame(fc_fuel, columns = ['fuelrate(kg/s)'])
    
#    fc_command = autonomie_out['fc_plant_cmd_simu']
#    fc_command_out = pd.DataFrame(fc_command, columns = ['fc_command_pwr(watt)'])
#    
#    fc_on = autonomie_out['fc_plant_on_simu']
#    fc_on_out = pd.DataFrame(fc_on, columns = ['fc_on'])
#    
#    fc_temp = autonomie_out['fc_plant_temp_coeff_simu']
#    fc_temp_out = pd.DataFrame(fc_temp, columns = ['fc_temp_coeff'])  
    
#    eng_fuel = autonomie_out['eng_plant_fuel_rate_simu']
#    eng_fuel_out = pd.DataFrame(eng_fuel, columns = ['fuelrate(kg/s)'])
##    
#    eng_on = autonomie_out['eng_plant_on_simu']
#    eng_on_out = pd.DataFrame(eng_on, columns = ['eng_on'])
#    
#    eng_speed = autonomie_out['eng_plant_spd_out_simu']
#    eng_speed_out = pd.DataFrame(eng_speed, columns = ['engine_speed(rad/s)'])
#    
#    eng_torque = autonomie_out['eng_plant_trq_out_simu']
#    eng_torque_out = pd.DataFrame(eng_torque, columns = ['engine_torque(Nm)'])
    
    elec_rate = autonomie_out['ess_plant_energy_in_simu']
    elec_rate_out = pd.DataFrame(elec_rate, columns = ['cum_elec(J)'])
    elec_rate_out['elec_energy(J)'] = elec_rate_out['cum_elec(J)'] - elec_rate_out['cum_elec(J)'].shift(1)
    elec_rate_out.loc[0, 'elec_energy(J)'] = 0
    road_grade = autonomie_out['env_sch_grade_simu']
    road_grade_out = pd.DataFrame(road_grade, columns = ['road_grade(rad)'])
    auxillary_power = autonomie_out['accelec_plant_pwr_simu']
    auxillary_power_out = pd.DataFrame(auxillary_power, columns = ['auxillary_power(watt)'])
    auxillary_power_value = auxillary_power_out.loc[0, 'auxillary_power(watt)']
    cycle_interporlate = autonomie_out['drv_lin_spd_dmd_simu']
    cycle_out = pd.DataFrame(cycle_interporlate, columns = ['Speed(m/s)'])
    cycle_out['Speed(mph)'] = cycle_out['Speed(m/s)'] * 2.23694
    cycle_out['Acceleration(mph/s)'] = cycle_out['Speed(mph)'] - cycle_out['Speed(mph)'].shift(1)
    cycle_out.loc[0, 'Acceleration(mph/s)'] = 0
    cycle_out.loc[:, 'Acceleration(mph/s)'] *= 10
    torque_demand = autonomie_out['drv_trq_dmd_simu']
    torque_demand_out = pd.DataFrame(torque_demand, columns = ['torque_demand(Nm)'])
    
    ess_current = autonomie_out['ess_plant_curr_out_simu']
    ess_current_out = pd.DataFrame(ess_current, columns = ['ess_current'])
    
    ess_volt = autonomie_out['ess_plant_volt_out_simu']
    ess_volt_out = pd.DataFrame(ess_volt, columns = ['ess_volt'])
    
    motor_command = autonomie_out['mot_ctrl_cmd_simu']
    motor_command_out = pd.DataFrame(motor_command, columns = ['mot_command'])
    
    motor_torque = autonomie_out['mot_plant_trq_out_simu']
    motor_torque_out = pd.DataFrame(motor_torque, columns = ['mot_torque(Nm)'])
    
    motor_speed = autonomie_out['mot_plant_spd_out_simu']
    motor_speed_out = pd.DataFrame(motor_speed, columns = ['mot_speed(rad/s)'])
    
#    motor2_command = autonomie_out['mot2_ctrl_cmd_simu']
#    motor2_command_out = pd.DataFrame(motor2_command, columns = ['mot2_command'])
#    
#    motor2_torque = autonomie_out['mot2_plant_trq_out_simu']
#    motor2_torque_out = pd.DataFrame(motor2_torque, columns = ['mot2_torque(Nm)'])
#    
#    motor2_speed = autonomie_out['mot2_plant_spd_out_simu']
#    motor2_speed_out = pd.DataFrame(motor2_speed, columns = ['mot2_speed(rad/s)'])
    trip_output = pd.concat([time_step_out, SOC_out, cycle_out, tractive_power_out, elec_rate_out, auxillary_power_out, road_grade_out, torque_demand_out, ess_current_out, ess_volt_out, motor_command_out, motor_torque_out, motor_speed_out], axis=1)
    trip_output['veh_weight'] = vehicle_weight_kg
    trip_output['battery_size(kWh)'] = battery_cap
    trip_output['SOC_max'] = soc_max
    trip_output['SOC_min'] = soc_min
    trip_output['SOC_target'] = soc_target
    if 'hwy' in str(cycle_name.item(0)):
        trip_output['road_type'] = 1
    elif 'local' in str(cycle_name.item(0)) :
        trip_output['road_type'] = 2
    else:
        trip_output['road_type'] = 0
#    trip_output['SOC_max'] = soc_max
    file_name = veh_type + '_' + str(cycle_name.item(0)) + '_'+ str(int(auxillary_power_value)) + '.csv'
    trip_output.to_csv(veh_type + '_hvac_csv/' + file_name)
    
#    break
        

# <codecell>
#plt.subplot(211)
#final_cycle_in = final_cycle_in.loc[(final_cycle_in['Acceleration(mph/s)']<=10) & (final_cycle_in['Acceleration(mph/s)']>=-10)]
##from scipy.stats import kendalltau
#sns.jointplot(final_cycle_in['Speed(mph)'], final_cycle_in['Acceleration(mph/s)'], kind="scatter", color="#4CB391", alpha = 0.3)   
#plt.savefig('speed_acc_dist1.jpg', dpi = 500) 
#plt.subplot(212)
#final_cycle_out = final_cycle_out.loc[(final_cycle_out['Acceleration(mph/s)']<=10) & (final_cycle_out['Acceleration(mph/s)']>=-10)]
##from scipy.stats import kendalltau
#sns.jointplot(final_cycle_out['Speed(mph)'], final_cycle_out['Acceleration(mph/s)'], kind="scatter", alpha = 0.3)    
#plt.savefig('speed_acc_dist2.jpg', dpi = 500)
#    plt.plot(cycle_interporlate)
#    plt.plot(raw_cycle)
#    break