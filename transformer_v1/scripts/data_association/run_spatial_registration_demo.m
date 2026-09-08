%% run_spatial_registration_demo.m
% 第五模块：空间配准
%
% 功能：
%   1. 读取第四模块输出 CleanTrack_OutlierRemoval.mat；
%   2. 读取 AlignedObs 中的 Radar / EO / ESM 相对量测和 UAV 位姿；
%   3. 将 Radar / EO / ESM 的平台相对量测统一转换到世界坐标系 XYZ；
%   4. 以第四模块 CleanTrack.pos_clean 作为世界坐标参考航迹；
%   5. 在世界坐标系下估计并补偿不同传感器的残余空间偏差；
%   6. 输出配准前后的世界坐标 XYZ；
%   7. 输出 RMSE / MAE / P95 / MaxError / Dispersion 等指标；
%   8. 绘制一张 RMSE 精度提升效果图。
%
% 输入文件：
%   CleanTrack_OutlierRemoval.mat
%
% 输出文件：
%   RegisteredTrack_SpatialRegistration.mat
%   SpatialRegistration_Metrics.mat
%   SpatialRegistration_Metrics.csv
%   SpatialRegistration_RMSE_Improvement.png
%
% 核心输出：
%   RegisteredTrack.Target(i).world_xyz_before
%   RegisteredTrack.Target(i).world_xyz_after
%   RegisteredTrack.Target(i).world_xyz_ref
%   RegisteredTrack.Target(i).world_xyz_eval

% % clear; clc; close all;

fprintf('\n===== 第五模块：空间配准 =====\n');

%% ========================================================================
% 1. 读取第四模块输出
% ========================================================================
if ~exist('CleanTrack_OutlierRemoval.mat', 'file')
    error('未找到 CleanTrack_OutlierRemoval.mat，请先运行第四模块 run_outlier_removal_demo.m');
end

load('CleanTrack_OutlierRemoval.mat');

if ~exist('CleanTrack', 'var')
    error('CleanTrack_OutlierRemoval.mat 中未找到变量 CleanTrack。');
end

if exist('AlignedObs', 'var')
    % 直接使用文件中保存的 AlignedObs
elseif isfield(CleanTrack, 'AlignedObs')
    AlignedObs = CleanTrack.AlignedObs;
else
    error('CleanTrack_OutlierRemoval.mat 中未找到 AlignedObs，无法进行世界坐标转换。');
end

fprintf('已读取第四模块输出：CleanTrack_OutlierRemoval.mat\n');

%% ========================================================================
% 2. 基本检查
% ========================================================================
if ~isfield(CleanTrack, 't_grid')
    error('CleanTrack 中缺少 t_grid。');
end

if ~isfield(CleanTrack, 'Target')
    error('CleanTrack 中缺少 Target。');
end

if ~isfield(AlignedObs, 'Radar') || ...
        ~isfield(AlignedObs, 'EO') || ...
        ~isfield(AlignedObs, 'ESM') || ...
        ~isfield(AlignedObs, 'truth')
    error('AlignedObs 中缺少 Radar / EO / ESM / truth 字段。');
end

if ~isfield(AlignedObs.truth, 'uav_pos_all') || ...
        ~isfield(AlignedObs.truth, 'uav_euler_all')
    error('AlignedObs.truth 中缺少 uav_pos_all 或 uav_euler_all，无法完成世界坐标转换。');
end

t_grid = CleanTrack.t_grid(:);
K = length(t_grid);

if isfield(CleanTrack, 'N_true')
    N_true = CleanTrack.N_true;
else
    N_true = length(CleanTrack.Target);
    CleanTrack.N_true = N_true;
end

fprintf('统一时间点数量 K = %d\n', K);
fprintf('目标数量 N_true = %d\n', N_true);

%% ========================================================================
% 3. 参数设置
% ========================================================================
RegParam = struct();

% 是否加入轻微演示用系统偏差。
% true：模拟坐标转换后仍存在的小尺度平台定位误差、姿态误差或安装误差；
% false：完全基于原始量测进行处理。
RegParam.enableDemoBias = true;

% 世界坐标系下的轻微系统偏差，单位 m。
% 注意：这里不应设置为几百米量级，否则空间配准会导致 RMSE 非真实地大幅下降。
RegParam.demoWorldBias.Radar = [ 35, -22,   8];
RegParam.demoWorldBias.EO    = [-28,  25,  -6];
RegParam.demoWorldBias.ESM   = [ 42,  18,  10];

% 在线空间偏差估计系数。
% 设置得小一些，保证空间配准只是缓慢、小幅修正。
RegParam.alphaRadar = 0.06;
RegParam.alphaEO    = 0.06;
RegParam.alphaESM   = 0.05;

% 单类传感器最大空间补偿幅值，单位 m。
% 用于限制配准强度，避免把观测点强行拉回参考航迹。
RegParam.maxBiasNorm = 60;

% 多源融合权重
RegParam.weightRadar = 1.00;
RegParam.weightEO    = 0.75;
RegParam.weightESM   = 0.55;

% 有真值时优先用真值评价，否则用 CleanTrack.pos_clean 评价
RegParam.useTruthForRMSE = true;

%% ========================================================================
% 4. 执行空间配准
% ========================================================================
[RegisteredTrack, RegistrationStats] = spatial_registration_core( ...
    CleanTrack, ...
    AlignedObs, ...
    RegParam);

RegisteredTrack.coordinate = 'world';
RegisteredTrack.unit = 'meter';
RegisteredTrack.output_description = ...
    ['world_xyz_before denotes the fused XYZ observations in the world coordinate system before spatial registration; ', ...
     'world_xyz_after denotes the registered XYZ trajectories in the world coordinate system; ', ...
     'world_xyz_ref denotes the reference track from CleanTrack in the world coordinate system; ', ...
     'world_xyz_eval denotes the evaluation trajectory, using ground truth when available.'];

fprintf('\n空间配准完成。\n');

fprintf('配准前总体 RMSE = %.3f m\n', RegistrationStats.rmse_before_total);
fprintf('配准后总体 RMSE = %.3f m\n', RegistrationStats.rmse_after_total);
fprintf('总体 RMSE 提升量 = %.3f m\n', RegistrationStats.rmse_improvement_total);
fprintf('总体 RMSE 改善率 = %.2f%%\n', 100 * RegistrationStats.improvementRate_total);

fprintf('配准前总体 MAE = %.3f m\n', RegistrationStats.mae_before_total);
fprintf('配准后总体 MAE = %.3f m\n', RegistrationStats.mae_after_total);

fprintf('配准前总体 P95 = %.3f m\n', RegistrationStats.p95_before_total);
fprintf('配准后总体 P95 = %.3f m\n', RegistrationStats.p95_after_total);

fprintf('配准前总体 MaxError = %.3f m\n', RegistrationStats.max_before_total);
fprintf('配准后总体 MaxError = %.3f m\n', RegistrationStats.max_after_total);

fprintf('配准前总体传感器离散度 = %.3f m\n', RegistrationStats.dispersion_before_total);
fprintf('配准后总体传感器离散度 = %.3f m\n', RegistrationStats.dispersion_after_total);
fprintf('总体离散度改善率 = %.2f%%\n', 100 * RegistrationStats.dispersionRate_total);

%% ========================================================================
% 5. 保存第五模块输出
% ========================================================================
save('RegisteredTrack_SpatialRegistration.mat', ...
    'RegisteredTrack', ...
    'RegistrationStats', ...
    'RegParam', ...
    '-v7.3');

fprintf('\n第五模块输出已保存：RegisteredTrack_SpatialRegistration.mat\n');

%% ========================================================================
% 6. 保存空间配准指标表
% ========================================================================
MetricTable = build_registration_metric_table(RegisteredTrack, RegistrationStats);

save('SpatialRegistration_Metrics.mat', ...
    'MetricTable', ...
    'RegistrationStats', ...
    '-v7.3');

writetable(MetricTable, 'SpatialRegistration_Metrics.csv');

fprintf('空间配准指标已保存：SpatialRegistration_Metrics.mat\n');
fprintf('空间配准指标表已保存：SpatialRegistration_Metrics.csv\n');

fprintf('\n===== 空间配准指标表 =====\n');
disp(MetricTable);

%% ========================================================================
% 7. 绘制 RMSE 精度提升效果图
% ========================================================================
plot_rmse_improvement_figure(MetricTable);

fprintf('空间配准 RMSE 提升效果图已保存：SpatialRegistration_RMSE_Improvement.png\n');

fprintf('\n===== 第五模块运行结束 =====\n');

%% ========================================================================
% 核心函数：世界坐标空间配准
% ========================================================================
