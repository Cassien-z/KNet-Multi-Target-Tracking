% batch_process_multitarget.m
% 批处理多目标数据，执行数据关联5大模块

clear; clc; close all;
addpath('../../src/data_association');

data_dir = '../../train_data_multitarget';
out_dir = '../../data/multitarget_data_registered';

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

files = dir(fullfile(data_dir, 'track_data_*.mat'));

fprintf('Found %d files to process.\n', length(files));

for f = 1:length(files)
    inFile = fullfile(data_dir, files(f).name);
    outFile = fullfile(out_dir, files(f).name);
    
    fprintf('\n========================================\n');
    fprintf('Processing [%d/%d]: %s\n', f, length(files), files(f).name);
    fprintf('========================================\n');
    
    try
        % 1. Time Align
        run('run_time_align_demo.m');
        
        % 2. Track Association
        run('run_track_association_demo.m');
        
        % 3. Track Completion
        run('run_track_completion_demo.m');
        
        % 4. Outlier Removal
        run('run_outlier_removal_demo.m');
        
        % 5. Spatial Registration
        run('run_spatial_registration_demo.m');
        
        % Rename final output
        movefile('RegisteredTrack_SpatialRegistration.mat', outFile);
        
        fprintf('Successfully processed and saved to %s\n', outFile);
    catch ME
        fprintf('Error processing %s: %s\n', files(f).name, ME.message);
        disp(ME.stack(1));
    end
end

fprintf('\nAll processing complete.\n');
