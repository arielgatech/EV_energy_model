

files = dir('C:\Users\X-Xu\Documents\TEMPO-Suite\EV_analysis\EV-analysis\Autonomie_batch_run\cycles\*.csv'); %directory to all the .csv cycles
for file = files' %loop through the cycle file
    raw_cycle = csvread(file.name, 1, 0); %load individual cycle
    sch_cycle = raw_cycle(:,[1,2]);
    sch_grade = raw_cycle(:,[1,3]);    
    [m,n] = size(sch_cycle); % size (duration) of the cycle
    disp(m);
    sch_key_on = [0 1; m-1 1];
%     max_grade = 0; % set max grade as 0
%     sch_grade = [0 0; m-1 max_grade];  % define grade array
%     sch_key_on = [0 1; m-1 1]; % define key-on array (stay key on)
    sch_metadata.name = strsplit(file.name,'.csv'); % define process name
    sch_metadata.proprietary = 'public'; %grant process use permission
    cd 'C:\Users\X-Xu\Documents\TEMPO-Suite\EV_analysis\EV-analysis\Autonomie_batch_run\cycles' % set directory for .mat files
    file_name = char(strcat(char(sch_metadata.name(1,1)),'.mat')); % define .mat file name
    save(sprintf('ARC_%s',file_name), 'sch_cycle', 'sch_grade', 'sch_key_on', 'sch_metadata');
    % save .mat files
%     break;
end

cd 'C:\Autonomie\Rev16SP7\Projects\default' %link to installed Autonomie
a_startup %start autonomie
cd 'C:\Autonomie\Rev16SP7\Program\gui\Matlab Files\model_manager' %Launch Autonomie process manage system
import_drive_cycles % load .mat file, it will be converted into Autonomie .process files
