%% run_time_align_demo.m
% 第一模块：时间尺度对齐主程序
%
% 功能：
%   1. 调用 generate_test_time_align_data.m 生成一体化原始异步数据；
%   2. 对 Radar / EO / ESM 的每条 localTrackID 做时间尺度对齐；
%   3. 时间对齐时不跨越长时间缺失段强行插值；
%   4. 保存 AlignedObs_TimeAlign.mat，作为第二模块航迹关联输入；
%   5. 保存所有无人机平台位姿，用于第二模块跨平台关联；
%   6. 输出展示图 TimeAlignment_Radar_LocalTrack.png。

% % clear; clc; close all;

fprintf('\n===== 第一模块：时间尺度对齐 =====\n');

%% ========================================================================
% 1. 生成并加载一体化原始数据
% ========================================================================

if exist('inFile', 'var')
    load(inFile);
    fprintf('\n已加载 %s\n', inFile);
else
    load('integrated_test_data.mat');
    fprintf('\n已加载 integrated_test_data.mat\n');
end


%% ========================================================================
% 2. 基本变量检查
% ========================================================================
if ~exist('Data_Input', 'var')
    error('integrated_test_data.mat 中缺少 Data_Input。');
end

if ~exist('truth', 'var')
    error('integrated_test_data.mat 中缺少 truth。');
end

if ~exist('t_grid', 'var')
    if isfield(truth, 't_grid')
        t_grid = truth.t_grid;
    else
        error('缺少 t_grid。');
    end
end

if ~isfield(truth, 'Nr') || ~isfield(truth, 'Ne') || ~isfield(truth, 'Nm')
    error('truth 中缺少 Nr / Ne / Nm。');
end

if ~isfield(truth, 'RadarSeg') || ~isfield(truth, 'EOSeg') || ~isfield(truth, 'ESMSeg')
    error('truth 中缺少 RadarSeg / EOSeg / ESMSeg。');
end

Nr = truth.Nr;
Ne = truth.Ne;
Nm = truth.Nm;

K_grid = length(t_grid);

fprintf('统一时间点数量 K_grid = %d\n', K_grid);
fprintf('Radar localTrack 数量 Nr = %d\n', Nr);
fprintf('EO localTrack 数量 Ne = %d\n', Ne);
fprintf('ESM localTrack 数量 Nm = %d\n', Nm);

%% ========================================================================
% 3. 初始化 AlignedObs
% ========================================================================
AlignedObs = struct();

AlignedObs.t_grid = t_grid(:);

AlignedObs.Radar.Y = nan(K_grid, Nr, 3);
AlignedObs.Radar.valid = false(K_grid, Nr);
AlignedObs.Radar.truthID = zeros(K_grid, Nr);
AlignedObs.Radar.uavID = zeros(1, Nr);

AlignedObs.EO.Y = nan(K_grid, Ne, 2);
AlignedObs.EO.valid = false(K_grid, Ne);
AlignedObs.EO.truthID = zeros(K_grid, Ne);
AlignedObs.EO.uavID = zeros(1, Ne);

AlignedObs.ESM.Y = nan(K_grid, Nm, 1);
AlignedObs.ESM.valid = false(K_grid, Nm);
AlignedObs.ESM.truthID = zeros(K_grid, Nm);
AlignedObs.ESM.uavID = zeros(1, Nm);

%% ========================================================================
% 4. 写入真值信息
% ========================================================================
if isfield(truth, 'N_true')
    AlignedObs.truth.N_true = truth.N_true;
else
    AlignedObs.truth.N_true = 6;
end

if isfield(truth, 'target_pos')
    AlignedObs.truth.target_pos = truth.target_pos;
end

if isfield(truth, 'uav_pos')
    AlignedObs.truth.uav_pos = truth.uav_pos;
end

if isfield(truth, 'uav_euler')
    AlignedObs.truth.uav_euler = truth.uav_euler;
end

if isfield(truth, 'targetRadarGapMask')
    AlignedObs.truth.targetRadarGapMask = truth.targetRadarGapMask;
end

if isfield(truth, 'uav_pos_raw')
    AlignedObs.truth.uav_pos_all_raw = truth.uav_pos_raw;
end

if isfield(truth, 'uav_euler_raw')
    AlignedObs.truth.uav_euler_all_raw = truth.uav_euler_raw;
end

if isfield(truth, 'uav_pos_raw') && ...
        isfield(truth, 'uav_euler_raw') && ...
        isfield(truth, 't_raw')

    nUAV_all = size(truth.uav_pos_raw, 2);
    K_grid_all = length(t_grid);

    AlignedObs.truth.uav_pos_all = zeros(K_grid_all, nUAV_all, 3);
    AlignedObs.truth.uav_euler_all = zeros(K_grid_all, nUAV_all, 3);

    for u = 1:nUAV_all
        for d = 1:3

            AlignedObs.truth.uav_pos_all(:, u, d) = interp1( ...
                truth.t_raw, ...
                squeeze(truth.uav_pos_raw(:, u, d)), ...
                t_grid, ...
                'pchip', ...
                'extrap');

            AlignedObs.truth.uav_euler_all(:, u, d) = interp1( ...
                truth.t_raw, ...
                squeeze(truth.uav_euler_raw(:, u, d)), ...
                t_grid, ...
                'pchip', ...
                'extrap');
        end
    end
end

AlignedObs.info.description = ...
    'Time-aligned Radar / EO / ESM local tracks generated from asynchronous measurements';

AlignedObs.info.radar_format = '[range, azimuth, elevation]';
AlignedObs.info.eo_format = '[azimuth, elevation]';
AlignedObs.info.esm_format = '[azimuth]';
AlignedObs.info.maxInterpGap = 0.45;

%% ========================================================================
% 5. Radar 时间尺度对齐
% ========================================================================
fprintf('\n--- Radar 时间尺度对齐 ---\n');

for localID = 1:Nr

    [hasSeg, trueID, activeMask] = local_get_segment_info( ...
        truth.RadarSeg, localID, t_grid);

    if ~hasSeg
        continue;
    end

    uavID = local_get_uav_id('Radar', localID);

    [t_meas, Y_meas_raw] = local_get_history_meas( ...
        Data_Input, uavID, trueID, 'Radar');

    if isempty(t_meas)
        continue;
    end

    [Y_align_raw, valid_align] = local_time_align_interp( ...
        t_meas, Y_meas_raw, t_grid, activeMask);

    Y_align = nan(K_grid, 3);
    Y_align(:, 1) = Y_align_raw(:, 3);
    Y_align(:, 2) = Y_align_raw(:, 1);
    Y_align(:, 3) = Y_align_raw(:, 2);

    AlignedObs.Radar.Y(:, localID, :) = Y_align;
    AlignedObs.Radar.valid(:, localID) = valid_align;
    AlignedObs.Radar.truthID(valid_align, localID) = trueID;
    AlignedObs.Radar.uavID(localID) = uavID;

    fprintf('Radar localID = %2d, targetID = %d, uavID = %d, valid = %d\n', ...
        localID, trueID, uavID, sum(valid_align));
end

%% ========================================================================
% 6. EO 时间尺度对齐
% ========================================================================
fprintf('\n--- EO 时间尺度对齐 ---\n');

for localID = 1:Ne

    [hasSeg, trueID, activeMask] = local_get_segment_info( ...
        truth.EOSeg, localID, t_grid);

    if ~hasSeg
        continue;
    end

    uavID = local_get_uav_id('EO', localID);

    [t_meas, Y_meas] = local_get_history_meas( ...
        Data_Input, uavID, trueID, 'EO');

    if isempty(t_meas)
        continue;
    end

    [Y_align, valid_align] = local_time_align_interp( ...
        t_meas, Y_meas, t_grid, activeMask);

    AlignedObs.EO.Y(:, localID, :) = Y_align;
    AlignedObs.EO.valid(:, localID) = valid_align;
    AlignedObs.EO.truthID(valid_align, localID) = trueID;
    AlignedObs.EO.uavID(localID) = uavID;

    fprintf('EO localID = %2d, targetID = %d, uavID = %d, valid = %d\n', ...
        localID, trueID, uavID, sum(valid_align));
end

%% ========================================================================
% 7. ESM 时间尺度对齐
% ========================================================================
fprintf('\n--- ESM 时间尺度对齐 ---\n');

for localID = 1:Nm

    [hasSeg, trueID, activeMask] = local_get_segment_info( ...
        truth.ESMSeg, localID, t_grid);

    if ~hasSeg
        continue;
    end

    uavID = local_get_uav_id('ESM', localID);

    [t_meas, Y_meas] = local_get_history_meas( ...
        Data_Input, uavID, trueID, 'ESM');

    if isempty(t_meas)
        continue;
    end

    Y_meas_az = Y_meas(:, 1);

    [Y_align_az, valid_align] = local_time_align_interp( ...
        t_meas, Y_meas_az, t_grid, activeMask);

    AlignedObs.ESM.Y(:, localID, 1) = Y_align_az;
    AlignedObs.ESM.valid(:, localID) = valid_align;
    AlignedObs.ESM.truthID(valid_align, localID) = trueID;
    AlignedObs.ESM.uavID(localID) = uavID;

    fprintf('ESM localID = %2d, targetID = %d, uavID = %d, valid = %d\n', ...
        localID, trueID, uavID, sum(valid_align));
end

%% ========================================================================
% 8. 保存时间尺度对齐结果
% ========================================================================
save('AlignedObs_TimeAlign.mat', 'AlignedObs');

fprintf('\n已保存：AlignedObs_TimeAlign.mat\n');

%% ========================================================================
% 9. 输出统计信息
% ========================================================================
numRadar = sum(AlignedObs.Radar.valid(:));
numEO    = sum(AlignedObs.EO.valid(:));
numESM   = sum(AlignedObs.ESM.valid(:));

fprintf('\n===== 时间尺度对齐统计 =====\n');
fprintf('Radar 对齐有效点数 = %d\n', numRadar);
fprintf('EO 对齐有效点数    = %d\n', numEO);
fprintf('ESM 对齐有效点数   = %d\n', numESM);

%% ========================================================================
% 10. 绘图：原始异步 Radar 量测与对齐结果
% ========================================================================
[~, idxLongest] = max(truth.RadarSeg(:, 4) - truth.RadarSeg(:, 3));
localID_plot = truth.RadarSeg(idxLongest, 1);

[hasSeg, trueID_plot, ~] = local_get_segment_info( ...
    truth.RadarSeg, localID_plot, t_grid);

if hasSeg

    uavID_plot = local_get_uav_id('Radar', localID_plot);

    [t_raw_radar, Y_raw_radar] = local_get_history_meas( ...
        Data_Input, uavID_plot, trueID_plot, 'Radar');

    Y_align_radar = squeeze(AlignedObs.Radar.Y(:, localID_plot, :));
    valid_align = AlignedObs.Radar.valid(:, localID_plot);

    figure('Color', 'w', 'Position', [200, 100, 920, 760]);

    subplot(3, 1, 1);
    hold on; grid on; box on;

    plot(t_raw_radar, Y_raw_radar(:, 1), ...
        'ko', 'MarkerSize', 4, 'LineWidth', 1.0);

    plot(t_grid(valid_align), Y_align_radar(valid_align, 2), ...
        'b-', 'LineWidth', 1.8);

    ylabel('方位角 / rad', 'FontName', 'SimHei');

    title(['Radar局部航迹', num2str(localID_plot), ...
        '的原始异步量测与时间尺度对齐结果'], ...
        'FontName', 'SimHei');

    legend('原始异步Radar量测', ...
           '时间尺度对齐结果', ...
           'Location', 'best', ...
           'FontName', 'SimHei');

    set(gca, 'FontName', 'SimHei');

    subplot(3, 1, 2);
    hold on; grid on; box on;

    plot(t_raw_radar, Y_raw_radar(:, 2), ...
        'ko', 'MarkerSize', 4, 'LineWidth', 1.0);

    plot(t_grid(valid_align), Y_align_radar(valid_align, 3), ...
        'b-', 'LineWidth', 1.8);

    ylabel('俯仰角 / rad', 'FontName', 'SimHei');

    legend('原始异步Radar量测', ...
           '时间尺度对齐结果', ...
           'Location', 'best', ...
           'FontName', 'SimHei');

    set(gca, 'FontName', 'SimHei');

    subplot(3, 1, 3);
    hold on; grid on; box on;

    plot(t_raw_radar, Y_raw_radar(:, 3), ...
        'ko', 'MarkerSize', 4, 'LineWidth', 1.0);

    plot(t_grid(valid_align), Y_align_radar(valid_align, 1), ...
        'b-', 'LineWidth', 1.8);

    xlabel('时间 / s', 'FontName', 'SimHei');
    ylabel('距离 / m', 'FontName', 'SimHei');

    legend('原始异步Radar量测', ...
           '时间尺度对齐结果', ...
           'Location', 'best', ...
           'FontName', 'SimHei');

    set(gca, 'FontName', 'SimHei');

    saveas(gcf, 'TimeAlignment_Radar_LocalTrack.png');

    fprintf('\n已保存展示图：TimeAlignment_Radar_LocalTrack.png\n');

else
    warning('未找到用于绘图的 Radar localID = %d。', localID_plot);
end

fprintf('\n===== run_time_align_demo 运行结束 =====\n');

%% ========================================================================
% 局部函数：读取局部航迹段信息
%% ========================================================================
function [hasSeg, trueID, activeMask] = local_get_segment_info(SegTable, localID, t_grid)

hasSeg = false;
trueID = NaN;
activeMask = false(size(t_grid));

if isempty(SegTable)
    return;
end

if istable(SegTable)
    SegTable = table2array(SegTable);
end

rows = SegTable(SegTable(:, 1) == localID, :);

if isempty(rows)
    return;
end

row = rows(1, :);

trueID = row(2);
startTime = row(3);
endTime = row(4);

activeMask = (t_grid >= startTime) & (t_grid <= endTime);

hasSeg = true;

end

%% ========================================================================
% 局部函数：根据 localID 给出 UAV 编号
%% ========================================================================
function uavID = local_get_uav_id(sensorName, localID)

switch sensorName

    case 'Radar'
        Radar_uav = [1 1 1 2 2 2 3 3 3 4 4 4 1 2 3 4];
        uavID = Radar_uav(localID);

    case 'EO'
        EO_uav = [1 1 1 4 4 4 5 5 5 1 4 5 1 4 5 5];
        uavID = EO_uav(localID);

    case 'ESM'
        ESM_uav = [1 1 1 3 3 3 5 5 5 1 3 5 1 3];
        uavID = ESM_uav(localID);

    otherwise
        error('未知传感器类型：%s', sensorName);
end

end

%% ========================================================================
% 局部函数：提取某一 UAV、某一目标、某类传感器的原始异步量测
%% ========================================================================
function [t_meas, Y_meas] = local_get_history_meas(Data_Input, uavID, targetID, sensorName)

t_meas = [];
Y_meas = [];

K = length(Data_Input);

for k = 1:K

    if ~isfield(Data_Input(k), 'UAV')
        continue;
    end

    if uavID > length(Data_Input(k).UAV)
        continue;
    end

    UAV = Data_Input(k).UAV(uavID);

    if ~isfield(UAV, sensorName)
        continue;
    end

    S = UAV.(sensorName);

    if ~isfield(S, 'Mask') || ~isfield(S, 'Meas') || ~isfield(S, 'MeasTime')
        continue;
    end

    if targetID > length(S.Mask)
        continue;
    end

    if S.Mask(targetID) ~= 1
        continue;
    end

    y = S.Meas(targetID, :);

    if isempty(y) || any(~isfinite(y))
        continue;
    end

    measTime = S.MeasTime;

    if numel(measTime) > 1
        measTime = measTime(targetID);
    end

    if isempty(measTime) || ~isfinite(measTime)
        continue;
    end

    t_meas(end+1, 1) = measTime; %#ok<AGROW>
    Y_meas(end+1, :) = y; %#ok<AGROW>
end

if ~isempty(t_meas)

    [t_meas, idx] = sort(t_meas);
    Y_meas = Y_meas(idx, :);

    [t_meas, ia] = unique(t_meas, 'stable');
    Y_meas = Y_meas(ia, :);
end

end

%% ========================================================================
% 局部函数：缺口感知时间尺度对齐
%% ========================================================================
function [Y_align, valid_align] = local_time_align_interp(t_meas, Y_meas, t_grid, activeMask)

t_grid = t_grid(:);
activeMask = activeMask(:);

if isvector(Y_meas)
    Y_meas = Y_meas(:);
end

K_grid = length(t_grid);
dimY = size(Y_meas, 2);

Y_align = nan(K_grid, dimY);
valid_align = false(K_grid, 1);

if isempty(t_meas) || size(Y_meas, 1) < 2
    return;
end

[t_meas, idx] = sort(t_meas(:));
Y_meas = Y_meas(idx, :);

[t_meas, ia] = unique(t_meas, 'stable');
Y_meas = Y_meas(ia, :);

if numel(t_meas) < 2
    return;
end

insideMask = (t_grid >= min(t_meas)) & (t_grid <= max(t_meas));

% 不跨越长缺失段插值。原始传感器更新周期约为 0.1~0.2 s。
maxInterpGap = 0.45;

nearestDist = inf(K_grid, 1);

for kk = 1:K_grid
    nearestDist(kk) = min(abs(t_meas - t_grid(kk)));
end

nearMeasMask = nearestDist <= maxInterpGap;

valid_align = activeMask & insideMask & nearMeasMask;

if sum(valid_align) == 0
    return;
end

for d = 1:dimY

    yd = Y_meas(:, d);

    good = isfinite(t_meas) & isfinite(yd);

    if sum(good) < 2
        continue;
    end

    if sum(good) >= 4
        interpMethod = 'pchip';
    else
        interpMethod = 'linear';
    end

    Y_align(valid_align, d) = interp1( ...
        t_meas(good), ...
        yd(good), ...
        t_grid(valid_align), ...
        interpMethod);
end

valid_dim = all(isfinite(Y_align), 2);
valid_align = valid_align & valid_dim;

Y_align(~valid_align, :) = NaN;

end