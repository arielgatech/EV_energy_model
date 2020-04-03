%Protected: false
%%Demo Function 4
%%Loop on Vehicles, Processes and parameters
%%Note: This example script should be run from a project folder.
%%Eg: C:\Autonomie\Rev15\Projects\default 
%%a_startup is a file available in the project folder.  
function resultsTable = XiaodanBatchSimulationRuns()

cd 'C:\Autonomie\Rev16SP7\Projects\default' %link to installed Autonomie
a_startup %start autonomie

vehicleList = {'elec_midsize_200er_dm.a_vehicle'}; % define vehicle model
files = dir('C:\Autonomie\Rev16SP7\Projects\default\processes\process\cycle\*.a_process'); % path to autonomie processes

% processList = {'ece.a_process',...
%    'accelerationUS.a_process'};
processList = {};
for file = files'
    if strfind(file.name, 'ABM')
        processList{end+1} = file.name;
    end
% end


% parameterMultiplierList = {'veh.plant.init.mass.total',1.39};
soc_list = {0.9};             
resultsTable{1,1} = 'Vehicle Name';
resultsTable{1,2} = 'Process Name';
resultsTable{1,3} = 'Vehicle Mass (kg)';
resultsTable{1,4} = 'Initial Battery level';
resultsTable{1,5} = 'Motor Power (watt)';
resultsTable{1,6} = 'Engine Power (watt)';
resultsTable{1,7} = 'Target SOC';

cnt = 2;
for j=1:length(vehicleList)
    simulationAsCell = command_load(vehicleList{j});
    auton_get_parameter_value_all('clear'); %%Clear the memory cache. Caching is done for speed.

    for i=1:length(processList)
        disp(processList{i})
        %init_soc_raw = strsplit(processList{i},'_');
        init_soc = 0.9; % for BEV, initial SOC is 90 percent, for other EVs, please use random number between max and min SOC
        disp(init_soc);
%         for m=1:length(soc_list)
        % for valueIdx = 2:size(parameterMultiplierList,2)
        %for paramIdx = 1:size(parameterMultiplierList,1)
        parameterName = 'ess.plant.init.soc_init';
        %disp(processList{i});
        %parameterOldValue = auton_get_parameter_value_all(simulationAsCell,parameterName);
        %parameterMulti = parameterMultiplierList{paramIdx,valueIdx};
        parameterNewValue = init_soc; % define SOC using custom input
        simulationAsCell = auton_set_parameter_value_all(simulationAsCell,parameterName,parameterNewValue);
        simulationWithProcAsCell = command_bind(simulationAsCell,processList{i});
        results = run_sandbox_simulation(simulationWithProcAsCell);
        resultsTable{cnt,1} = vehicleList{j};
        resultsTable{cnt,2} = processList{i};
        resultsTable{cnt,3} = get_datafile_result_formatted(results,'veh.plant.init.mass.total');
        resultsTable{cnt,4} = get_datafile_result_formatted(results,'ess.plant.init.soc_init');
        resultsTable{cnt,5} = get_datafile_result_formatted(results,'mot.plant.scale.pwr_max_des');
        resultsTable{cnt,6} = 85000;
        resultsTable{cnt,7} = 0.6;
        cnt = cnt + 1;
%         end
    end
auton_cellTable2XlsFile('BEV200_Results.xlsx',resultsTable);%%Write to a file, this table is not used.  The raw output can be found in Autonomie Result folder directly    
end

write_cell_table(1,resultsTable);%%Write to the screen.
% auton_cellTable2XlsFile('300MILE_BEV_Results.xlsx',resultsTable);%%Write to a file.
