%% run_outlier_removal_demo.m
% 第四模块：野值剔除
%
% 功能：
%   1. 读取第三模块输出 OutlierInput_OutlierRemoval.mat；
%   2. 若该文件不存在，则自动从 CompletedTrack_TrackCompletion.mat 构造 OutlierInput；
%   3. 对补全后的目标级三维航迹进行野值检测；
%   4. 对检测出的野值点进行局部插值修正；
%   5. 输出清洗后的目标级航迹 CleanTrack；
%   6. 保留 OutlierInput / CompletionInput / AlignedObs，供第五模块空间配准使用；
%   7. 绘制三维野值剔除结果图；
%   8. 绘制各目标 RMSE 提升量图。
%
% 输入文件：
%   OutlierInput_OutlierRemoval.mat
%
% 兼容输入：
%   CompletedTrack_TrackCompletion.mat
%   CompletionInput_TrackCompletion.mat
%   AlignedObs_TimeAlign.mat
%
% 输出文件：
%   CleanTrack_OutlierRemoval.mat
%
% 输出图片：
%   OutlierRemoval_3D_Result.png
%   OutlierRemoval_RMSE_Improvement.png

% % clear; clc; close all;

set(0, 'DefaultFigureVisible', 'on');
set(0, 'DefaultFigureWindowStyle', 'normal');
set(groot, 'DefaultFigureVisible', 'on');
set(groot, 'DefaultFigureWindowStyle', 'normal');

fprintf('\n===== 第四模块：野值剔除 =====\n');

%% ========================================================================
% 1. 读取第三模块输出
% ========================================================================
OutlierInput = [];

if exist('OutlierInput_OutlierRemoval.mat', 'file')

    load('OutlierInput_OutlierRemoval.mat');

    if ~exist('OutlierInput', 'var')
        error('OutlierInput_OutlierRemoval.mat 中未找到变量 OutlierInput。');
    end

    fprintf('已读取：OutlierInput_OutlierRemoval.mat\n');

elseif exist('CompletedTrack_TrackCompletion.mat', 'file')

    load('CompletedTrack_TrackCompletion.mat');

    if ~exist('CompletedTrack', 'var')
        error('CompletedTrack_TrackCompletion.mat 中未找到变量 CompletedTrack。');
    end

    OutlierInput = struct();
    OutlierInput.CompletedTrack = CompletedTrack;

    if isfield(CompletedTrack, 't_grid')
        OutlierInput.t_grid = CompletedTrack.t_grid(:);
    else
        error('CompletedTrack 中未找到 t_grid。');
    end

    if exist('CompletionStats', 'var')
        OutlierInput.CompletionStats = CompletionStats;
    end

    if exist('CompletionInput_TrackCompletion.mat', 'file')

        temp = load('CompletionInput_TrackCompletion.mat');

        if isfield(temp, 'CompletionInput')

            OutlierInput.CompletionInput = temp.CompletionInput;

            if isfield(temp.CompletionInput, 'AlignedObs')
                OutlierInput.AlignedObs = temp.CompletionInput.AlignedObs;
            end
        end
    end

    save('OutlierInput_OutlierRemoval.mat', ...
        'OutlierInput', ...
        '-v7.3');

    fprintf('未找到 OutlierInput_OutlierRemoval.mat，已自动构造并保存。\n');

else

    error(['未找到 OutlierInput_OutlierRemoval.mat 或 CompletedTrack_TrackCompletion.mat，', ...
        '请先运行第三模块 run_track_completion_demo.m。']);
end

%% ========================================================================
% 2. 提取 CompletedTrack / CompletionInput / AlignedObs
% ========================================================================
if ~isfield(OutlierInput, 'CompletedTrack')
    error('OutlierInput 中未找到 CompletedTrack。');
end

CompletedTrack = OutlierInput.CompletedTrack;

if isfield(OutlierInput, 't_grid')
    t_grid = OutlierInput.t_grid(:);
elseif isfield(CompletedTrack, 't_grid')
    t_grid = CompletedTrack.t_grid(:);
else
    error('无法解析 t_grid。');
end

K = length(t_grid);

if isfield(CompletedTrack, 'N_true')
    N_true = CompletedTrack.N_true;
else
    N_true = length(CompletedTrack.Target);
end

CompletionInput = [];

if isfield(OutlierInput, 'CompletionInput')
    CompletionInput = OutlierInput.CompletionInput;
elseif exist('CompletionInput_TrackCompletion.mat', 'file')

    temp = load('CompletionInput_TrackCompletion.mat');

    if isfield(temp, 'CompletionInput')
        CompletionInput = temp.CompletionInput;
    end
end

AlignedObs = [];

if isfield(OutlierInput, 'AlignedObs')
    AlignedObs = OutlierInput.AlignedObs;
elseif ~isempty(CompletionInput) && isfield(CompletionInput, 'AlignedObs')
    AlignedObs = CompletionInput.AlignedObs;
elseif isfield(CompletedTrack, 'AlignedObs')
    AlignedObs = CompletedTrack.AlignedObs;
elseif exist('AlignedObs_TimeAlign.mat', 'file')

    temp = load('AlignedObs_TimeAlign.mat');

    if isfield(temp, 'AlignedObs')
        AlignedObs = temp.AlignedObs;
    end
end

if isempty(AlignedObs)
    warning('未解析到 AlignedObs。第五模块坐标转换将无法运行。');
else
    fprintf('已解析 AlignedObs，并将在第四模块输出中继续保存。\n');
end

fprintf('统一时间点数量 K = %d\n', K);
fprintf('目标数量 N_true = %d\n', N_true);

%% ========================================================================
% 3. 调用野值剔除核心函数
% ========================================================================
[CleanTrack, OutlierStats] = outlier_removal_simple(CompletedTrack, t_grid);

fprintf('\n野值剔除完成。\n');
fprintf('总有效点数量 = %d\n', OutlierStats.totalValid);
fprintf('检测野值数量 = %d\n', OutlierStats.totalOutlier);
fprintf('修正野值数量 = %d\n', OutlierStats.totalCorrected);
fprintf('野值比例 = %.2f%%\n', 100 * OutlierStats.outlierRatio);

if isfinite(OutlierStats.rmse_before_total)
    fprintf('剔除前总体 RMSE = %.3f m\n', OutlierStats.rmse_before_total);
end

if isfinite(OutlierStats.rmse_after_total)
    fprintf('剔除后总体 RMSE = %.3f m\n', OutlierStats.rmse_after_total);
end

if isfinite(OutlierStats.rmse_improvement_total)
    fprintf('总体 RMSE 提升量 = %.3f m\n', OutlierStats.rmse_improvement_total);
end

if isfinite(OutlierStats.improvementRate_total)
    fprintf('总体 RMSE 改善率 = %.2f%%\n', 100 * OutlierStats.improvementRate_total);
end

%% ========================================================================
% 4. 保留前序信息，方便第五模块使用
% ========================================================================
CleanTrack.OutlierInput = OutlierInput;

if ~isempty(CompletionInput)
    CleanTrack.CompletionInput = CompletionInput;
end

if ~isempty(AlignedObs)
    CleanTrack.AlignedObs = AlignedObs;
end

OutlierOutput = struct();
OutlierOutput.CleanTrack = CleanTrack;
OutlierOutput.OutlierStats = OutlierStats;
OutlierOutput.OutlierInput = OutlierInput;

if ~isempty(CompletionInput)
    OutlierOutput.CompletionInput = CompletionInput;
end

if ~isempty(AlignedObs)
    OutlierOutput.AlignedObs = AlignedObs;
end

%% ========================================================================
% 5. 保存第四模块输出
% ========================================================================
save('CleanTrack_OutlierRemoval.mat', ...
    'CleanTrack', ...
    'OutlierStats', ...
    'OutlierOutput', ...
    'OutlierInput', ...
    'CompletionInput', ...
    'AlignedObs', ...
    '-v7.3');

fprintf('\n第四模块输出已保存：CleanTrack_OutlierRemoval.mat\n');

%% ========================================================================
% 6. 图1：三维野值剔除结果
% ========================================================================
fig1 = figure( ...
    'Name', '三维野值剔除结果', ...
    'NumberTitle', 'off', ...
    'Visible', 'on', ...
    'WindowStyle', 'normal', ...
    'Color', 'w', ...
    'Position', [100 80 1200 780]);

ax1 = axes(fig1);
hold(ax1, 'on');
grid(ax1, 'on');
box(ax1, 'on');

colorSet = lines(N_true);

hTrue = [];
hBefore = [];
hAfter = [];
hOutlier = [];

allXYZ = [];

for targetID = 1:N_true

    T = CleanTrack.Target(targetID);
    c = colorSet(targetID, :);

    pos_before = T.pos_before_outlier_removal;
    pos_after = T.pos_clean;
    pos_true = T.pos_true;

    valid_before = T.valid_before_outlier_removal;
    valid_after = T.valid_clean;
    outlierMask = T.outlierMask;

    validTruth = all(isfinite(pos_true), 2);

    if any(validTruth)

        h = plot3(ax1, ...
            pos_true(validTruth,1), ...
            pos_true(validTruth,2), ...
            pos_true(validTruth,3), ...
            'k--', ...
            'LineWidth', 1.0);

        if isempty(hTrue)
            hTrue = h;
        else
            set(h, 'HandleVisibility', 'off');
        end

        allXYZ = [allXYZ; pos_true(validTruth,:)]; %#ok<AGROW>
    end

    if any(valid_before)

        h = plot3(ax1, ...
            pos_before(valid_before,1), ...
            pos_before(valid_before,2), ...
            pos_before(valid_before,3), ...
            ':', ...
            'Color', c, ...
            'LineWidth', 1.2);

        if isempty(hBefore)
            hBefore = h;
        else
            set(h, 'HandleVisibility', 'off');
        end

        allXYZ = [allXYZ; pos_before(valid_before,:)]; %#ok<AGROW>
    end

    if any(valid_after)

        h = plot3(ax1, ...
            pos_after(valid_after,1), ...
            pos_after(valid_after,2), ...
            pos_after(valid_after,3), ...
            '-', ...
            'Color', c, ...
            'LineWidth', 1.8);

        if isempty(hAfter)
            hAfter = h;
        else
            set(h, 'HandleVisibility', 'off');
        end

        allXYZ = [allXYZ; pos_after(valid_after,:)]; %#ok<AGROW>
    end

    if any(outlierMask)

        h = scatter3(ax1, ...
            pos_before(outlierMask,1), ...
            pos_before(outlierMask,2), ...
            pos_before(outlierMask,3), ...
            55, ...
            'r', ...
            'x', ...
            'LineWidth', 1.5);

        if isempty(hOutlier)
            hOutlier = h;
        else
            set(h, 'HandleVisibility', 'off');
        end

        allXYZ = [allXYZ; pos_before(outlierMask,:)]; %#ok<AGROW>
    end
end

xlabel(ax1, 'X / m');
ylabel(ax1, 'Y / m');
zlabel(ax1, 'Z / m');

title(ax1, '三维航迹野值剔除结果', ...
    'FontWeight', 'normal');

view(ax1, 42, 24);
axis(ax1, 'vis3d');

local_set_axis_range(ax1, allXYZ);

set(ax1, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'GridAlpha', 0.25, ...
    'Projection', 'perspective');

legendHandles = [];
legendNames = {};

if ~isempty(hTrue)
    legendHandles = [legendHandles, hTrue]; %#ok<AGROW>
    legendNames{end+1} = '真实轨迹参考'; %#ok<AGROW>
end

if ~isempty(hBefore)
    legendHandles = [legendHandles, hBefore]; %#ok<AGROW>
    legendNames{end+1} = '剔除前航迹'; %#ok<AGROW>
end

if ~isempty(hAfter)
    legendHandles = [legendHandles, hAfter]; %#ok<AGROW>
    legendNames{end+1} = '剔除后航迹'; %#ok<AGROW>
end

if ~isempty(hOutlier)
    legendHandles = [legendHandles, hOutlier]; %#ok<AGROW>
    legendNames{end+1} = '检测野值点'; %#ok<AGROW>
end

if ~isempty(legendHandles)
    legend(ax1, legendHandles, legendNames, 'Location', 'eastoutside');
end

drawnow;
figure(fig1);
shg;
pause(0.2);

exportgraphics(fig1, 'OutlierRemoval_3D_Result.png', ...
    'Resolution', 300);

fprintf('已导出：OutlierRemoval_3D_Result.png\n');

%% ========================================================================
% 7. 图2：各目标 RMSE 提升量
%
% 说明：
%   RMSE提升量 = RMSE_before - RMSE_after。
%   数值越大，说明野值剔除后的误差改善越明显。
% ========================================================================
fig2 = figure( ...
    'Name', '野值剔除RMSE提升量', ...
    'NumberTitle', 'off', ...
    'Visible', 'on', ...
    'WindowStyle', 'normal', ...
    'Color', 'w', ...
    'Position', [160 130 920 470]);

rmseBefore = nan(N_true, 1);
rmseAfter = nan(N_true, 1);
rmseImprove = nan(N_true, 1);
improveRate = nan(N_true, 1);

for targetID = 1:N_true

    rmseBefore(targetID) = CleanTrack.Target(targetID).rmse_before;
    rmseAfter(targetID) = CleanTrack.Target(targetID).rmse_after;

    if isfinite(rmseBefore(targetID)) && isfinite(rmseAfter(targetID))

        rmseImprove(targetID) = rmseBefore(targetID) - rmseAfter(targetID);

        if rmseBefore(targetID) > 0
            improveRate(targetID) = rmseImprove(targetID) / rmseBefore(targetID);
        end
    end
end

bar(rmseImprove);

grid on;
box on;

xlabel('目标编号');
ylabel('RMSE提升量 / m');
title('野值剔除后的各目标RMSE提升量');

set(gca, 'XTick', 1:N_true);
set(gca, 'XTickLabel', arrayfun(@num2str, 1:N_true, 'UniformOutput', false));

validImprove = rmseImprove(isfinite(rmseImprove));

if isempty(validImprove)

    ylim([0, 1]);

else

    minVal = min(validImprove);
    maxVal = max(validImprove);

    if minVal == maxVal
        if minVal >= 0
            ylim([0, max(1, 1.2 * maxVal)]);
        else
            ylim([1.2 * minVal, 0]);
        end
    else
        yLow = min(0, 1.20 * minVal);
        yHigh = max(1, 1.20 * maxVal);
        ylim([yLow, yHigh]);
    end
end

hold on;

yl = ylim;
yRange = yl(2) - yl(1);

for targetID = 1:N_true

    if ~isfinite(rmseImprove(targetID))
        continue;
    end

    if rmseImprove(targetID) >= 0
        yText = rmseImprove(targetID) + 0.03 * yRange;
        vAlign = 'bottom';
    else
        yText = rmseImprove(targetID) - 0.03 * yRange;
        vAlign = 'top';
    end

    if isfinite(improveRate(targetID))
        labelText = sprintf('%.1f m\n%.1f%%', ...
            rmseImprove(targetID), ...
            100 * improveRate(targetID));
    else
        labelText = sprintf('%.1f m', rmseImprove(targetID));
    end

    text(targetID, yText, ...
        labelText, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', vAlign, ...
        'FontSize', 8);
end

yline(0, 'k-', 'LineWidth', 1.0);

drawnow;
figure(fig2);
shg;
pause(0.2);

exportgraphics(fig2, 'OutlierRemoval_RMSE_Improvement.png', ...
    'Resolution', 300);

fprintf('已导出：OutlierRemoval_RMSE_Improvement.png\n');

%% ========================================================================
% 8. 命令行输出每个目标统计
% ========================================================================
fprintf('\n===== 各目标野值剔除统计 =====\n');

for targetID = 1:N_true

    fprintf(['目标 %d: 有效点=%d, 野值点=%d, 修正点=%d, ', ...
        'RMSE_before=%.3f m, RMSE_after=%.3f m, 提升量=%.3f m, 改善率=%.2f%%\n'], ...
        targetID, ...
        CleanTrack.Target(targetID).numValid, ...
        CleanTrack.Target(targetID).numOutlier, ...
        CleanTrack.Target(targetID).numCorrected, ...
        CleanTrack.Target(targetID).rmse_before, ...
        CleanTrack.Target(targetID).rmse_after, ...
        CleanTrack.Target(targetID).rmse_improvement, ...
        100 * CleanTrack.Target(targetID).improvementRate);
end

fprintf('\n===== 第四模块运行结束 =====\n');

%% ========================================================================
% 局部函数：野值剔除核心函数
% ========================================================================
