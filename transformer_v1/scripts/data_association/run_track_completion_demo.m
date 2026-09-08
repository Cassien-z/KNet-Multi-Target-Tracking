%% run_track_completion_demo.m
% 第三模块：数据补全
%
% 输入：
%   CompletionInput_TrackCompletion.mat
%
% 输出：
%   CompletedTrack_TrackCompletion.mat
%   OutlierInput_OutlierRemoval.mat
%
% 关键改动：
%   1. 保存 OutlierInput_OutlierRemoval.mat；
%   2. OutlierInput 中保留 CompletionInput 和 AlignedObs；
%   3. 确保第四模块和第五模块可以继续读取前序数据链。

% % clear; clc; close all;

set(0, 'DefaultFigureVisible', 'on');
set(0, 'DefaultFigureWindowStyle', 'normal');

fprintf('\n===== 第三模块：数据补全 =====\n');

%% ========================================================================
% 1. 读取第二模块输出
% ========================================================================
if ~exist('CompletionInput_TrackCompletion.mat', 'file')
    error('未找到 CompletionInput_TrackCompletion.mat，请先运行 run_track_association_demo.m');
end

load('CompletionInput_TrackCompletion.mat');

if ~exist('CompletionInput', 'var')
    error('CompletionInput_TrackCompletion.mat 中未找到变量 CompletionInput。');
end

if ~isfield(CompletionInput, 'AlignedObs')
    error('CompletionInput 中未找到 AlignedObs。请检查第二模块是否保存 AlignedObs。');
end

AlignedObs = CompletionInput.AlignedObs;
t_grid = CompletionInput.t_grid(:);

K = length(t_grid);

fprintf('已读取 CompletionInput_TrackCompletion.mat\n');
fprintf('统一时间点数量 K = %d\n', K);

%% ========================================================================
% 2. 数据补全
% ========================================================================
[CompletedTrack, CompletionStats] = track_completion_simple(CompletionInput);

fprintf('\n数据补全完成。\n');
fprintf('目标数量 = %d\n', CompletedTrack.N_true);
fprintf('原始有效点数量 = %d\n', CompletionStats.totalRaw);
fprintf('内部缺失点数量 = %d\n', CompletionStats.totalMissingInternal);
fprintf('成功补全点数量 = %d\n', CompletionStats.totalFilled);
fprintf('补全覆盖率 = %.2f%%\n', 100 * CompletionStats.fillCoverage);

%% ========================================================================
% 3. 保存第三模块输出
% ========================================================================
save('CompletedTrack_TrackCompletion.mat', ...
    'CompletedTrack', ...
    'CompletionStats', ...
    '-v7.3');

fprintf('已保存：CompletedTrack_TrackCompletion.mat\n');

%% ========================================================================
% 4. 构造并保存第四模块输入
% ========================================================================
OutlierInput = struct();

OutlierInput.t_grid = t_grid;
OutlierInput.CompletedTrack = CompletedTrack;
OutlierInput.CompletionStats = CompletionStats;
OutlierInput.CompletionInput = CompletionInput;
OutlierInput.AlignedObs = AlignedObs;

save('OutlierInput_OutlierRemoval.mat', ...
    'OutlierInput', ...
    '-v7.3');

fprintf('已保存：OutlierInput_OutlierRemoval.mat\n');

%% ========================================================================
% 5. 绘图：三维补全结果
% ========================================================================
fig1 = figure( ...
    'Name', '三维航迹补全结果', ...
    'NumberTitle', 'off', ...
    'Visible', 'on', ...
    'Color', 'w', ...
    'Position', [100 80 1200 780]);

ax = axes(fig1);
hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');

N_true = CompletedTrack.N_true;
colorSet = lines(N_true);

allXYZ = [];

for targetID = 1:N_true

    T = CompletedTrack.Target(targetID);
    c = colorSet(targetID, :);

    pos_true = T.pos_true;
    pos_raw = T.pos_raw;
    pos_completed = T.pos_completed;

    valid_raw = T.valid_raw;
    valid_completed = T.valid_completed;
    filledMask = T.filledMask;

    validTruth = all(isfinite(pos_true), 2);

    if any(validTruth)
        plot3(pos_true(validTruth,1), pos_true(validTruth,2), pos_true(validTruth,3), ...
            '--', 'Color', c, 'LineWidth', 1.0);
        allXYZ = [allXYZ; pos_true(validTruth,:)]; %#ok<AGROW>
    end

    if any(valid_completed)
        plot3(pos_completed(valid_completed,1), pos_completed(valid_completed,2), pos_completed(valid_completed,3), ...
            '-', 'Color', c, 'LineWidth', 1.8);
        allXYZ = [allXYZ; pos_completed(valid_completed,:)]; %#ok<AGROW>
    end

    if any(valid_raw)
        scatter3(pos_raw(valid_raw,1), pos_raw(valid_raw,2), pos_raw(valid_raw,3), ...
            18, c, 'filled');
        allXYZ = [allXYZ; pos_raw(valid_raw,:)]; %#ok<AGROW>
    end

    if any(filledMask)
        scatter3(pos_completed(filledMask,1), pos_completed(filledMask,2), pos_completed(filledMask,3), ...
            45, 'k', 'filled', 'd');
        allXYZ = [allXYZ; pos_completed(filledMask,:)]; %#ok<AGROW>
    end
end

xlabel('X / m');
ylabel('Y / m');
zlabel('Z / m');
title('三维航迹补全结果');

view(42, 24);
axis vis3d;
local_set_axis_range(gca, allXYZ);

legend('真实轨迹', '补全后航迹', '原始有效点', '补全点', ...
    'Location', 'eastoutside');

exportgraphics(fig1, 'Track_Completion_3D_Result.png', 'Resolution', 300);
fprintf('已导出：Track_Completion_3D_Result.png\n');

%% ========================================================================
% 6. 绘图：各目标补全点 RMSE
% ========================================================================
fig2 = figure( ...
    'Name', '各目标补全点RMSE', ...
    'NumberTitle', 'off', ...
    'Visible', 'on', ...
    'Color', 'w', ...
    'Position', [160 130 900 450]);

rmseFilled = nan(N_true, 1);
numFilled = zeros(N_true, 1);

for targetID = 1:N_true
    rmseFilled(targetID) = CompletedTrack.Target(targetID).rmse_filled;
    numFilled(targetID) = CompletedTrack.Target(targetID).numFilled;
end

bar(rmseFilled);
grid on; box on;

xlabel('目标编号');
ylabel('补全点 RMSE / m');
title('各目标补全点误差');

set(gca, 'XTick', 1:N_true);

validRMSE = rmseFilled(isfinite(rmseFilled));

if isempty(validRMSE)
    ylim([0, 1]);
else
    ylim([0, 1.2 * max(validRMSE)]);
end

hold on;

for targetID = 1:N_true
    if isfinite(rmseFilled(targetID))
        text(targetID, rmseFilled(targetID), ...
            sprintf('%.1f\n补%d', rmseFilled(targetID), numFilled(targetID)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 8);
    end
end

exportgraphics(fig2, 'Track_Completion_Target_RMSE.png', 'Resolution', 300);
fprintf('已导出：Track_Completion_Target_RMSE.png\n');

fprintf('\n===== 第三模块运行结束 =====\n');

%% ========================================================================
% 局部函数：数据补全核心
% ========================================================================
function [CompletedTrack, CompletionStats] = track_completion_simple(CompletionInput)

AlignedObs = CompletionInput.AlignedObs;
t_grid = CompletionInput.t_grid(:);
K = length(t_grid);

if isfield(AlignedObs, 'truth') && isfield(AlignedObs.truth, 'N_true')
    N_true = AlignedObs.truth.N_true;
elseif isfield(AlignedObs, 'truth') && isfield(AlignedObs.truth, 'target_pos')
    N_true = size(AlignedObs.truth.target_pos, 2);
else
    error('无法确定目标数量 N_true。');
end

CompletedTrack = struct();
CompletedTrack.t_grid = t_grid;
CompletedTrack.N_true = N_true;
CompletedTrack.AlignedObs = AlignedObs;
CompletedTrack.description = ...
    'Target-level track completion output with AlignedObs retained';

Target = repmat(struct(), N_true, 1);

totalRaw = 0;
totalMissingInternal = 0;
totalFilled = 0;
totalCompleted = 0;

allFilledErr2 = [];
allCompletedErr2 = [];

for targetID = 1:N_true

    pos_raw = nan(K, 3);
    valid_raw = false(K, 1);

    for k = 1:K

        pList = [];

        Nr = size(AlignedObs.Radar.Y, 2);

        for rid = 1:Nr

            if ~AlignedObs.Radar.valid(k, rid)
                continue;
            end

            if AlignedObs.Radar.truthID(k, rid) ~= targetID
                continue;
            end

            y = squeeze(AlignedObs.Radar.Y(k, rid, :)).';

            if numel(y) < 3 || any(~isfinite(y))
                continue;
            end

            range = y(1);
            az = y(2);
            el = y(3);

            uavID = get_track_uav_id(AlignedObs.Radar, rid, 1);

            pUAV = local_get_uav_pos(AlignedObs, k, uavID);
            euler = local_get_uav_euler(AlignedObs, k, uavID);

            pWorld = local_radar_meas_to_world(range, az, el, pUAV, euler);

            if all(isfinite(pWorld))
                pList(end+1, :) = pWorld; %#ok<AGROW>
            end
        end

        if ~isempty(pList)
            pos_raw(k, :) = median(pList, 1);
            valid_raw(k) = true;
        end
    end

    pos_true = nan(K, 3);

    if isfield(AlignedObs, 'truth') && isfield(AlignedObs.truth, 'target_pos')
        pos_true = squeeze(AlignedObs.truth.target_pos(:, targetID, :));
    end

    pos_completed = pos_raw;
    valid_completed = valid_raw;

    validIdx = find(valid_raw);

    internalMissingMask = false(K, 1);

    if numel(validIdx) >= 2
        internalMissingMask(validIdx(1):validIdx(end)) = true;
        internalMissingMask = internalMissingMask & ~valid_raw;
    end

    filledMask = false(K, 1);

    if any(internalMissingMask)

        for d = 1:3

            y = pos_raw(:, d);
            good = valid_raw & isfinite(y);

            if sum(good) >= 4
                method = 'pchip';
            elseif sum(good) >= 2
                method = 'linear';
            else
                continue;
            end

            pos_completed(internalMissingMask, d) = interp1( ...
                t_grid(good), ...
                y(good), ...
                t_grid(internalMissingMask), ...
                method);
        end

        filledMask = internalMissingMask & all(isfinite(pos_completed), 2);
        valid_completed(filledMask) = true;
    end

    validTruth = all(isfinite(pos_true), 2);

    idxRaw = valid_raw & validTruth;
    idxFilled = filledMask & validTruth;
    idxCompleted = valid_completed & validTruth;

    if any(idxRaw)
        errRaw = pos_raw(idxRaw, :) - pos_true(idxRaw, :);
        rmse_raw = sqrt(mean(sum(errRaw.^2, 2)));
    else
        rmse_raw = NaN;
    end

    if any(idxFilled)
        errFilled = pos_completed(idxFilled, :) - pos_true(idxFilled, :);
        rmse_filled = sqrt(mean(sum(errFilled.^2, 2)));
        allFilledErr2 = [allFilledErr2; sum(errFilled.^2, 2)]; %#ok<AGROW>
    else
        rmse_filled = NaN;
    end

    if any(idxCompleted)
        errCompleted = pos_completed(idxCompleted, :) - pos_true(idxCompleted, :);
        rmse_completed = sqrt(mean(sum(errCompleted.^2, 2)));
        allCompletedErr2 = [allCompletedErr2; sum(errCompleted.^2, 2)]; %#ok<AGROW>
    else
        rmse_completed = NaN;
    end

    Target(targetID).targetID = targetID;

    Target(targetID).pos_raw = pos_raw;
    Target(targetID).valid_raw = valid_raw;

    Target(targetID).pos_completed = pos_completed;
    Target(targetID).valid_completed = valid_completed;

    Target(targetID).pos_true = pos_true;

    Target(targetID).internalMissingMask = internalMissingMask;
    Target(targetID).filledMask = filledMask;

    Target(targetID).numRaw = sum(valid_raw);
    Target(targetID).numMissingInternal = sum(internalMissingMask);
    Target(targetID).numFilled = sum(filledMask);
    Target(targetID).numCompleted = sum(valid_completed);

    Target(targetID).rmse_raw = rmse_raw;
    Target(targetID).rmse_filled = rmse_filled;
    Target(targetID).rmse_completed = rmse_completed;

    totalRaw = totalRaw + Target(targetID).numRaw;
    totalMissingInternal = totalMissingInternal + Target(targetID).numMissingInternal;
    totalFilled = totalFilled + Target(targetID).numFilled;
    totalCompleted = totalCompleted + Target(targetID).numCompleted;
end

CompletedTrack.Target = Target;

CompletionStats.totalRaw = totalRaw;
CompletionStats.totalMissingInternal = totalMissingInternal;
CompletionStats.totalFilled = totalFilled;
CompletionStats.totalCompleted = totalCompleted;
CompletionStats.fillCoverage = safe_div(totalFilled, totalMissingInternal);

if isempty(allFilledErr2)
    CompletionStats.rmse_filled = NaN;
else
    CompletionStats.rmse_filled = sqrt(mean(allFilledErr2));
end

if isempty(allCompletedErr2)
    CompletionStats.rmse_completed = NaN;
else
    CompletionStats.rmse_completed = sqrt(mean(allCompletedErr2));
end

end

%% ========================================================================
% 局部函数
% ========================================================================
function uavID = get_track_uav_id(SensorStruct, localID, defaultID)

uavID = defaultID;

if isfield(SensorStruct, 'uavID') && localID <= length(SensorStruct.uavID)
    tempID = SensorStruct.uavID(localID);
    if isfinite(tempID) && tempID >= 1
        uavID = tempID;
    end
end

end

function pUAV = local_get_uav_pos(AlignedObs, k, uavID)

if isfield(AlignedObs, 'truth') && isfield(AlignedObs.truth, 'uav_pos_all')
    nUAV = size(AlignedObs.truth.uav_pos_all, 2);
    uavID = min(max(round(uavID), 1), nUAV);
    pUAV = squeeze(AlignedObs.truth.uav_pos_all(k, uavID, :)).';
elseif isfield(AlignedObs, 'truth') && isfield(AlignedObs.truth, 'uav_pos')
    pUAV = AlignedObs.truth.uav_pos(k, :);
else
    pUAV = [0, 0, 0];
end

end

function euler = local_get_uav_euler(AlignedObs, k, uavID)

if isfield(AlignedObs, 'truth') && isfield(AlignedObs.truth, 'uav_euler_all')
    nUAV = size(AlignedObs.truth.uav_euler_all, 2);
    uavID = min(max(round(uavID), 1), nUAV);
    euler = squeeze(AlignedObs.truth.uav_euler_all(k, uavID, :)).';
elseif isfield(AlignedObs, 'truth') && isfield(AlignedObs.truth, 'uav_euler')
    euler = AlignedObs.truth.uav_euler(k, :);
else
    euler = [0, 0, 0];
end

end

function pWorld = local_radar_meas_to_world(range, az, el, pUAV, euler)

pBody = [
    range * cos(el) * cos(az);
    range * cos(el) * sin(az);
    range * sin(el)
    ];

Rwb = local_rpy2rot(euler);

pWorld = pUAV(:) + Rwb * pBody;
pWorld = pWorld(:).';

end

function R = local_rpy2rot(euler)

roll = euler(1);
pitch = euler(2);
yaw = euler(3);

Rx = [1 0 0;
      0 cos(roll) -sin(roll);
      0 sin(roll) cos(roll)];

Ry = [cos(pitch) 0 sin(pitch);
      0 1 0;
      -sin(pitch) 0 cos(pitch)];

Rz = [cos(yaw) -sin(yaw) 0;
      sin(yaw) cos(yaw) 0;
      0 0 1];

R = Rz * Ry * Rx;

end

function local_set_axis_range(ax, allXYZ)

if isempty(allXYZ)
    return;
end

allXYZ = allXYZ(all(isfinite(allXYZ), 2), :);

if isempty(allXYZ)
    return;
end

xMin = min(allXYZ(:,1)); xMax = max(allXYZ(:,1));
yMin = min(allXYZ(:,2)); yMax = max(allXYZ(:,2));
zMin = min(allXYZ(:,3)); zMax = max(allXYZ(:,3));

xPad = 0.06 * max(xMax - xMin, 1);
yPad = 0.06 * max(yMax - yMin, 1);
zPad = 0.10 * max(zMax - zMin, 1);

xlim(ax, [xMin - xPad, xMax + xPad]);
ylim(ax, [yMin - yPad, yMax + yPad]);
zlim(ax, [zMin - zPad, zMax + zPad]);

end

function r = safe_div(a, b)

if b == 0 || ~isfinite(b)
    r = NaN;
else
    r = a / b;
end

end