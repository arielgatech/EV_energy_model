# EV energy model
A scalable electric vehicle energy model toolkit for transportation networks
### Developer
Xiaodan Xu
### License
GNU General Public License v3.0
### Disclaimer
This work was supported by the National Center for Sustainable Transportation, a National University Transportation Center sponsored by the U.S. Department of Transportation (DOT 69A3551747114), and data and/or support were provided by the Georgia Department of Transportation and the Atlanta Regional Commission. The information, data, and/or work presented herein were funded in part by an agency of the United States Government, state government, and/or local government.  Neither the United States Government, state government, local government, nor any agencies thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights.  Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government, state government, local government, or any agencies thereof.  The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government, state government, local government, or any agency thereof.  The materials presented herein do not constitute not constitute a standard, specification, or regulation. The research team also acknowledges and appreciates Argonne National Laboratory for providing the Autonomie model and the Atlanta Regional Commission for providing data and technical support.

### Reference
Xu, X., H M Abdul Aziz, H. Liu, M.O. Rodgers, and R. Guensler (submitted, revised, in the third review). A Scalable Energy Modeling Framework for Electric Vehicles in Regional Transportation Networks. Applied Energy.

# Tutorial
Major functions in this repository include:
- **Autonomie_batch_run**: data generation with Autonomie
- **EV_model_training_with_bayesian_network**: training EV energy models with Bayesian Network method
- **example**: apply trained EV model to predict energy consumption of given cycle
- **EV_energy_rate_final.csv**: readily available EV energy rates for network applications
 
### Autonomie_batch_run
1. Use "gen_mat.m" to create Autonomie processes with local driving cycle csv files
2. Use "allvehBatchSimulationRuns.m" to run Autonomie with a specific EV model and all the processes
3. Use "allvehBatchSimulationRuns_HVAC.m" to run Autonomie for HVAC adjustment factors
4. Use "post_process_autonomie_py36.py" to post-process Autonomie simulation results

### EV_model_training_with_bayesian_network
1. Put post-processed autonomie training results into {vehicle_type}_csv as input
2. Run "{vehicle_type}_training_py36ver.py" to obtain trained parameters for each vehicle type
 
### example
Put second-by-second EV driving profile under a separate folder as input, in this example, the folder is called 'TESTING_CYCLE'
Update following parameters in the `main()` function in "example_cycle_energy_prediction.py":
```sh
def main():
    start_time = time.time() 
    path = "TESTING_CYCLE"  # path to input driving cycle
    hvac_load = 3 # define HVAC load in kw
    veh_type = "SER_PHEV"# define vehicle type, 'BEV100' -> 100-mile nissan leaf, 'BEV300' -> 300-mile tesla, 
    #PAR_HEV -> Ford fusion,  SER_PHEV -> Chevy volt, PS_PHEV -> Prius prime, PS_HEV -> Prius
    out_path = veh_type + '_AC/' # define output directory
```
The output results can be located at 'SER_PHEV_AC' folder in this case, you can also change the output directory to switch to another local foler

### EV_energy_rate_final.csv
ready-to-use energy consumption rate from this analysis
- **roadTypeID**: 1- highway, 2- local
- **speedBinID**: average roadway speed in mph, in 1 mph increment
- **initSOC**: initial battery state-of-charge (SOC) 
- **road_grade**: road grade in percentage value [-10%, 10%]
- **fuel_rate**: fuel (gasoline) consumption rate in kJ/mile
- **elec_rate** : electricity consumption rate from on-board battery in kJ/mile
- **vehType**: electric vehicle type, BEV- battery electric vehicle, FCEV - fuel-cell eletric vehicle, HEV - hybrid electric vehicle, PHEV - Plug-in hybrid electric vehicle
- **Powertrain**: powertrain specfication of each EV, including 100-mile BEV, 300-mile BEV, FCEV, series PHEV, power-split PHEV (PS_PHEV), power-split HEV (PS_HEV) and parallel HEV
