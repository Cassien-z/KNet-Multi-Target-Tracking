%% run_track_association_demo.m
% 第二模块：航迹关联主程序
%
% 功能：
%   1. 读取第一模块输出 AlignedObs_TimeAlign.mat；
%   2. 根据 AlignedObs 自动生成航迹关联矩阵；
%   3. 完成 Radar / EO / ESM 航迹关联；
%   4. 统计航迹关联正确率；
%   5. 保存第二模块输出 AssocOut_TrackAssociation.mat；
%   6. 导出第三模块“数据补全”的输入 CompletionInput_TrackCompletion.mat；
%   7. 绘制三维航迹关联结果图；
%   8. 绘制航迹关联性能统计图。
%
% 输出文件：
%   1) TrackAssocMatrix_TimeAlign.mat
%   2) AssocOut_TrackAssociation.mat
%   3) CompletionInput_TrackCompletion.mat
%
% 输出图片：
%   1) Track_Association_3D_Result.png
%   2) Track_Association_Performance.png
%
% 运行前：
%   run_time_align_demo
%
% 运行：
%   clear functions
%   rehash toolboxcache
%   run_track_association_demo

% % clear; clc; close all;

%% ========================================================================
% 强制图窗正常显示
% ========================================================================
set(0, 'DefaultFigureVisible', 'on');
set(0, 'DefaultFigureWindowStyle', 'normal');
set(groot, 'DefaultFigureVisible', 'on');
set(groot, 'DefaultFigureWindowStyle', 'normal');

fprintf('\n===== 第二模块：航迹关联 =====\n');

%% ========================================================================
% 0. 检查当前运行文件路径，避免调用到其他同名文件
% ========================================================================
fprintf('\n当前运行文件路径检查：\n');
which run_track_association_demo -all
which track_association_simple -all

%% ========================================================================
% 1. 读取第一模块输出
% ========================================================================
if ~exist('AlignedObs_TimeAlign.mat', 'file')
    error('未找到 AlignedObs_TimeAlign.mat，请先运行 run_time_align_demo.m');
end

load('AlignedObs_TimeAlign.mat');

if ~exist('AlignedObs', 'var')
    error('AlignedObs_TimeAlign.mat 中未找到变量 AlignedObs。');
end

t_grid = AlignedObs.t_grid(:);
K = length(t_grid);

Nr = size(AlignedObs.Radar.Y, 2);
Ne = size(AlignedObs.EO.Y, 2);
Nm = size(AlignedObs.ESM.Y, 2);

fprintf('\n已读取第一模块输出：AlignedObs_TimeAlign.mat\n');
fprintf('统一时间点数量 K = %d\n', K);
fprintf('Radar 局部航迹数量 = %d\n', Nr);
fprintf('EO 局部航迹数量    = %d\n', Ne);
fprintf('ESM 局部航迹数量   = %d\n', Nm);
fprintf('局部航迹总数       = %d\n', Nr + Ne + Nm);

%% ========================================================================
% 2. 构造与 AlignedObs 匹配的航迹关联矩阵，并调用航迹关联核心函数
%
% 说明：
%   AssocMatrix(i,j) = 1 表示第 i 条局部航迹与第 j 条局部航迹允许关联；
%   AssocMatrix(i,j) = 0 表示不允许关联。
%
% 全局编号规则：
%   Radar: 1 ~ Nr
%   EO:    Nr+1 ~ Nr+Ne
%   ESM:   Nr+Ne+1 ~ Nr+Ne+Nm
%
% 默认约束：
%   1. 自身不能关联；
%   2. 同源航迹不能关联；
%   3. 无有效点航迹不能参与关联；
%   4. 不同源之间默认允许关联。
% ========================================================================

AssocParam = build_assoc_matrix_from_alignedobs(AlignedObs);

% 将矩阵也挂到 AlignedObs 中，便于核心函数和后续模块读取
AlignedObs.AssocMatrix = AssocParam.AssocMatrix;
AlignedObs.AssocParam = AssocParam;

save('TrackAssocMatrix_TimeAlign.mat', 'AssocParam');

fprintf('\n已生成航迹关联矩阵：TrackAssocMatrix_TimeAlign.mat\n');
fprintf('航迹关联矩阵尺寸 = %d × %d\n', ...
    size(AssocParam.AssocMatrix, 1), ...
    size(AssocParam.AssocMatrix, 2));

fprintf('航迹全局编号规则：\n');
fprintf('  Radar: 1 ~ %d\n', Nr);
fprintf('  EO:    %d ~ %d\n', Nr + 1, Nr + Ne);
fprintf('  ESM:   %d ~ %d\n', Nr + Ne + 1, Nr + Ne + Nm);

AssocOut = track_association_simple(AlignedObs, AssocParam);

fprintf('\n航迹关联计算完成。\n');

%% ========================================================================
% 3. 统计航迹关联正确率
%
% 指标定义：
%   航迹关联正确率 = 正确关联对数量 / 输出关联对总数量
%
% 判断准则：
%   Radar-EO  正确：radarTruthID == eoTruthID
%   Radar-ESM 正确：radarTruthID == esmTruthID
% ========================================================================
num_RE_pair = 0;
num_RE_correct = 0;

num_RM_pair = 0;
num_RM_correct = 0;

for k = 1:K

    tracks = AssocOut.Frame(k).tracks;

    for i = 1:length(tracks)

        rTruth = tracks(i).radarTruthID;

        % Radar-EO
        if ~isnan(tracks(i).eoID)

            num_RE_pair = num_RE_pair + 1;

            if ~isnan(rTruth) && ~isnan(tracks(i).eoTruthID) && ...
                    rTruth == tracks(i).eoTruthID

                num_RE_correct = num_RE_correct + 1;
            end
        end

        % Radar-ESM
        if ~isnan(tracks(i).esmID)

            num_RM_pair = num_RM_pair + 1;

            if ~isnan(rTruth) && ~isnan(tracks(i).esmTruthID) && ...
                    rTruth == tracks(i).esmTruthID

                num_RM_correct = num_RM_correct + 1;
            end
        end
    end
end

rate_RE_pair = safe_div(num_RE_correct, num_RE_pair);
rate_RM_pair = safe_div(num_RM_correct, num_RM_pair);

rate_total_pair = safe_div( ...
    num_RE_correct + num_RM_correct, ...
    num_RE_pair + num_RM_pair);

fprintf('\n===== 航迹关联正确率统计 =====\n');
fprintf('Radar-EO  正确关联率 = %.2f%% (%d / %d)\n', ...
    100*rate_RE_pair, num_RE_correct, num_RE_pair);
fprintf('Radar-ESM 正确关联率 = %.2f%% (%d / %d)\n', ...
    100*rate_RM_pair, num_RM_correct, num_RM_pair);
fprintf('总体正确关联率      = %.2f%% (%d / %d)\n', ...
    100*rate_total_pair, ...
    num_RE_correct + num_RM_correct, ...
    num_RE_pair + num_RM_pair);

if rate_total_pair >= 0.85
    fprintf('\n结论：总体航迹关联正确率满足 85%% 的指标要求。\n');
else
    fprintf('\n结论：总体航迹关联正确率低于 85%%，需要进一步调整关联门限或代价函数。\n');
end

%% ========================================================================
% 4. 保存第二模块输出
% ========================================================================
AssocStats.rate_RE_pair = rate_RE_pair;
AssocStats.rate_RM_pair = rate_RM_pair;
AssocStats.rate_total_pair = rate_total_pair;

AssocStats.num_RE_correct = num_RE_correct;
AssocStats.num_RE_pair = num_RE_pair;

AssocStats.num_RM_correct = num_RM_correct;
AssocStats.num_RM_pair = num_RM_pair;

AssocStats.num_total_correct = num_RE_correct + num_RM_correct;
AssocStats.num_total_pair = num_RE_pair + num_RM_pair;

AssocStats.AssocParam = AssocParam;

save('AssocOut_TrackAssociation.mat', 'AssocOut', 'AssocStats');

fprintf('\n第二模块输出已保存：AssocOut_TrackAssociation.mat\n');

%% ========================================================================
% 5. 额外导出第三模块“数据补全”的输入数据
%
% 说明：
%   CompletionInput 是第三模块的标准输入。
%   其中同时包含：
%   1) 第一模块输出的时间对齐量测 AlignedObs；
%   2) 第二模块输出的航迹关联结果 AssocOut；
%   3) 第二模块统计指标 AssocStats；
%   4) 航迹关联矩阵 AssocParam；
%   5) 统一时间轴 t_grid；
%   6) Radar / EO / ESM 局部航迹数量。
% ========================================================================
CompletionInput.t_grid = t_grid;

CompletionInput.AlignedObs = AlignedObs;
CompletionInput.AssocOut = AssocOut;
CompletionInput.AssocStats = AssocStats;
CompletionInput.AssocParam = AssocParam;

CompletionInput.Nr = Nr;
CompletionInput.Ne = Ne;
CompletionInput.Nm = Nm;
CompletionInput.K = K;

save('CompletionInput_TrackCompletion.mat', 'CompletionInput');

fprintf('第三模块输入已保存：CompletionInput_TrackCompletion.mat\n');

%% ========================================================================
% 6. 打印某一时刻的关联结果
% ========================================================================
tShow = 30;
[~, kShow] = min(abs(t_grid - tShow));

fprintf('\n===== t = %.2f s 时刻的关联结果 =====\n', t_grid(kShow));

tracksShow = AssocOut.Frame(kShow).tracks;

if isempty(tracksShow)
    fprintf('当前时刻没有形成关联航迹。\n');
else
    for i = 1:length(tracksShow)

        fprintf('\n关联组 globalID = %d\n', tracksShow(i).globalID);
        fprintf('  Radar ID = %d, truthID = %g\n', ...
            tracksShow(i).radarID, tracksShow(i).radarTruthID);

        if isnan(tracksShow(i).eoID)
            fprintf('  EO ID    = 无关联\n');
        else
            fprintf('  EO ID    = %d, truthID = %g, costEO = %.6f\n', ...
                tracksShow(i).eoID, ...
                tracksShow(i).eoTruthID, ...
                tracksShow(i).costEO);
        end

        if isnan(tracksShow(i).esmID)
            fprintf('  ESM ID   = 无关联\n');
        else
            fprintf('  ESM ID   = %d, truthID = %g, costESM = %.6f\n', ...
                tracksShow(i).esmID, ...
                tracksShow(i).esmTruthID, ...
                tracksShow(i).costESM);
        end
    end
end

%% ========================================================================
% 7. 图1：三维航迹关联结果
%
% 图示逻辑：
%   1. 同一真实目标的 Radar 局部航迹片段使用相同颜色；
%   2. 不同 localTrackID 之间保持断开；
%   3. 虚线表示对应真实目标轨迹；
%   4. 圆点表示该时刻跨传感器关联与目标真值一致；
%   5. 叉号表示该时刻跨传感器关联存在不一致。
% ========================================================================

fig1 = figure( ...
    'Name', '三维航迹关联结果', ...
    'NumberTitle', 'off', ...
    'Visible', 'on', ...
    'WindowStyle', 'normal', ...
    'Color', 'w', ...
    'Position', [80 60 1250 820]);

ax1 = axes(fig1);
hold(ax1, 'on');
grid(ax1, 'on');
box(ax1, 'on');

if isfield(AlignedObs, 'truth') && isfield(AlignedObs.truth, 'N_true')
    N_true = AlignedObs.truth.N_true;
elseif isfield(AlignedObs, 'truth') && isfield(AlignedObs.truth, 'target_pos')
    N_true = size(AlignedObs.truth.target_pos, 2);
else
    N_true = max(AlignedObs.Radar.truthID(:));
end

colorSetTruth = lines(N_true);

hRadarSeg = [];
hTruthRef = [];
hConsistent = [];
hInconsistent = [];

allXYZ = [];

%% ------------------------------------------------------------------------
% 7.1 画真实目标轨迹
%% ------------------------------------------------------------------------
if isfield(AlignedObs, 'truth') && isfield(AlignedObs.truth, 'target_pos')

    target_pos = AlignedObs.truth.target_pos;
    N_true_plot = size(target_pos, 2);

    for n = 1:N_true_plot

        px = squeeze(target_pos(:, n, 1));
        py = squeeze(target_pos(:, n, 2));
        pz = squeeze(target_pos(:, n, 3));

        c = colorSetTruth(n, :);

        h = plot3(ax1, px, py, pz, '--', ...
            'Color', c, ...
            'LineWidth', 1.1);

        allXYZ = [allXYZ; px(:), py(:), pz(:)]; %#ok<AGROW>

        if isempty(hTruthRef)
            hTruthRef = h;
        else
            set(h, 'HandleVisibility', 'off');
        end
    end
end

%% ------------------------------------------------------------------------
% 7.2 按真实目标编号上色，画 Radar 三维局部航迹片段
%% ------------------------------------------------------------------------
for rid = 1:Nr

    valid = AlignedObs.Radar.valid(:, rid);

    if ~any(valid)
        continue;
    end

    truthList = AlignedObs.Radar.truthID(valid, rid);
    truthList = truthList(truthList > 0);

    if isempty(truthList)
        continue;
    end

    trueID = mode(truthList);

    idx = find(valid);
    pWorld = nan(length(idx), 3);

    for ii = 1:length(idx)

        k = idx(ii);

        rMeas = squeeze(AlignedObs.Radar.Y(k, rid, :)).';

        range = rMeas(1);
        az    = rMeas(2);
        el    = rMeas(3);

        pUAV  = AlignedObs.truth.uav_pos(k, :);
        euler = AlignedObs.truth.uav_euler(k, :);

        pWorld(ii, :) = local_radar_meas_to_world(range, az, el, pUAV, euler);
    end

    c = colorSetTruth(trueID, :);

    h = plot3(ax1, pWorld(:,1), pWorld(:,2), pWorld(:,3), '-o', ...
        'Color', c, ...
        'LineWidth', 1.8, ...
        'MarkerSize', 3.5, ...
        'MarkerFaceColor', c);

    allXYZ = [allXYZ; pWorld]; %#ok<AGROW>

    if isempty(hRadarSeg)
        hRadarSeg = h;
    else
        set(h, 'HandleVisibility', 'off');
    end
end

%% ------------------------------------------------------------------------
% 7.3 提取关联一致性标记点
%% ------------------------------------------------------------------------
sampleStep = 10;

P_consistent = [];
P_inconsistent = [];

for k = 1:K

    tracks = AssocOut.Frame(k).tracks;

    for i = 1:length(tracks)

        if isempty(tracks(i).radarMeas)
            continue;
        end

        hasAssoc = false;
        isConsistent = true;

        rTruth = tracks(i).radarTruthID;

        if ~isnan(tracks(i).eoID)
            hasAssoc = true;

            if isnan(rTruth) || isnan(tracks(i).eoTruthID) || ...
                    rTruth ~= tracks(i).eoTruthID

                isConsistent = false;
            end
        end

        if ~isnan(tracks(i).esmID)
            hasAssoc = true;

            if isnan(rTruth) || isnan(tracks(i).esmTruthID) || ...
                    rTruth ~= tracks(i).esmTruthID

                isConsistent = false;
            end
        end

        if ~hasAssoc
            continue;
        end

        rMeas = tracks(i).radarMeas;

        range = rMeas(1);
        az    = rMeas(2);
        el    = rMeas(3);

        pUAV  = AlignedObs.truth.uav_pos(k, :);
        euler = AlignedObs.truth.uav_euler(k, :);

        p = local_radar_meas_to_world(range, az, el, pUAV, euler);

        if isConsistent
            if mod(k, sampleStep) == 0
                P_consistent(end+1, :) = p; %#ok<AGROW>
            end
        else
            P_inconsistent(end+1, :) = p; %#ok<AGROW>
        end
    end
end

if ~isempty(P_consistent)
    hConsistent = scatter3(ax1, ...
        P_consistent(:,1), ...
        P_consistent(:,2), ...
        P_consistent(:,3), ...
        30, ...
        'k', ...
        'filled', ...
        'Marker', 'o');

    allXYZ = [allXYZ; P_consistent]; %#ok<AGROW>
end

if ~isempty(P_inconsistent)
    hInconsistent = scatter3(ax1, ...
        P_inconsistent(:,1), ...
        P_inconsistent(:,2), ...
        P_inconsistent(:,3), ...
        90, ...
        'r', ...
        'x', ...
        'LineWidth', 2.0);

    allXYZ = [allXYZ; P_inconsistent]; %#ok<AGROW>
end

%% ------------------------------------------------------------------------
% 7.4 坐标轴、美化与导出
%% ------------------------------------------------------------------------
xlabel(ax1, 'X / m', 'FontSize', 12);
ylabel(ax1, 'Y / m', 'FontSize', 12);
zlabel(ax1, 'Z / m', 'FontSize', 12);

title(ax1, '三维航迹关联结果', ...
    'FontSize', 14, ...
    'FontWeight', 'normal');

view(ax1, 42, 24);
axis(ax1, 'vis3d');

if ~isempty(allXYZ)

    xMin = min(allXYZ(:,1)); xMax = max(allXYZ(:,1));
    yMin = min(allXYZ(:,2)); yMax = max(allXYZ(:,2));
    zMin = min(allXYZ(:,3)); zMax = max(allXYZ(:,3));

    xPad = 0.06 * max(xMax - xMin, 1);
    yPad = 0.06 * max(yMax - yMin, 1);
    zPad = 0.10 * max(zMax - zMin, 1);

    xlim(ax1, [xMin - xPad, xMax + xPad]);
    ylim(ax1, [yMin - yPad, yMax + yPad]);
    zlim(ax1, [zMin - zPad, zMax + zPad]);
end

set(ax1, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'GridAlpha', 0.25, ...
    'Projection', 'perspective');

set(ax1, 'Position', [0.07 0.10 0.72 0.82]);

legendHandles = [];
legendNames = {};

if ~isempty(hRadarSeg)
    legendHandles = [legendHandles, hRadarSeg]; %#ok<AGROW>
    legendNames{end+1} = '雷达局部航迹片段'; %#ok<AGROW>
end

if ~isempty(hTruthRef)
    legendHandles = [legendHandles, hTruthRef]; %#ok<AGROW>
    legendNames{end+1} = '真实轨迹参考'; %#ok<AGROW>
end

if ~isempty(hConsistent)
    legendHandles = [legendHandles, hConsistent]; %#ok<AGROW>
    legendNames{end+1} = '一致关联'; %#ok<AGROW>
end

if ~isempty(hInconsistent)
    legendHandles = [legendHandles, hInconsistent]; %#ok<AGROW>
    legendNames{end+1} = '不一致关联'; %#ok<AGROW>
end

if ~isempty(legendHandles)
    lgd = legend(ax1, legendHandles, legendNames, ...
        'Location', 'eastoutside');

    set(lgd, ...
        'FontSize', 10, ...
        'Box', 'on');
end

drawnow;
figure(fig1);
shg;
pause(0.2);

exportgraphics(fig1, 'Track_Association_3D_Result.png', ...
    'Resolution', 300);

fprintf('已导出图片：Track_Association_3D_Result.png\n');

%% ========================================================================
% 8. 图2：航迹关联性能统计
%% ========================================================================

fig2 = figure( ...
    'Name', '航迹关联性能统计', ...
    'NumberTitle', 'off', ...
    'Visible', 'on', ...
    'WindowStyle', 'normal', ...
    'Color', 'w', ...
    'Position', [150 150 820 440]);

figure(fig2);

rateBar = 100 * [rate_RE_pair, rate_RM_pair, rate_total_pair];

bar(rateBar);

grid on; box on;

set(gca, 'XTickLabel', {'Radar-EO', 'Radar-ESM', 'Overall'});

ylabel('Correct Association Rate / %');
ylim([0, 105]);

title('航迹关联性能统计');

hold on;
yline(85, 'r--', 'LineWidth', 1.5);
text(2.25, 87, '85% 指标线', 'Color', 'r');

for i = 1:length(rateBar)
    text(i, rateBar(i) + 2, sprintf('%.1f%%', rateBar(i)), ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 10);
end

drawnow;
figure(fig2);
shg;
pause(0.2);

exportgraphics(fig2, 'Track_Association_Performance.png', ...
    'Resolution', 300);

fprintf('已导出图片：Track_Association_Performance.png\n');

drawnow;
fprintf('\n===== 第二模块运行结束 =====\n');

%% ========================================================================
% 局部函数：根据 AlignedObs 自动构造航迹关联矩阵
%% ========================================================================
function AssocParam = build_assoc_matrix_from_alignedobs(AlignedObs)

Nr = size(AlignedObs.Radar.Y, 2);
Ne = size(AlignedObs.EO.Y, 2);
Nm = size(AlignedObs.ESM.Y, 2);

Ntotal = Nr + Ne + Nm;

idxRadar = 1:Nr;
idxEO    = Nr + (1:Ne);
idxESM   = Nr + Ne + (1:Nm);

AssocMatrix = true(Ntotal, Ntotal);

% 自身不能关联
AssocMatrix(1:Ntotal+1:end) = false;

% 同源不能关联
AssocMatrix(idxRadar, idxRadar) = false;
AssocMatrix(idxEO, idxEO)       = false;
AssocMatrix(idxESM, idxESM)     = false;

% 不同源默认允许关联
AssocMatrix(idxRadar, idxEO)  = true;
AssocMatrix(idxEO, idxRadar)  = true;

AssocMatrix(idxRadar, idxESM) = true;
AssocMatrix(idxESM, idxRadar) = true;

AssocMatrix(idxEO, idxESM)    = true;
AssocMatrix(idxESM, idxEO)    = true;

% 无有效量测的局部航迹不能参与关联
radarValidTrack = squeeze(any(AlignedObs.Radar.valid, 1));
eoValidTrack    = squeeze(any(AlignedObs.EO.valid, 1));
esmValidTrack   = squeeze(any(AlignedObs.ESM.valid, 1));

allValidTrack = [radarValidTrack(:); eoValidTrack(:); esmValidTrack(:)];

for i = 1:Ntotal
    if ~allValidTrack(i)
        AssocMatrix(i, :) = false;
        AssocMatrix(:, i) = false;
    end
end

% 保证对称
AssocMatrix = AssocMatrix & AssocMatrix.';

trackName = cell(Ntotal, 1);
sourceType = zeros(Ntotal, 1);
localID = zeros(Ntotal, 1);

for i = 1:Nr
    gid = idxRadar(i);
    trackName{gid} = sprintf('Radar-%d', i);
    sourceType(gid) = 1;
    localID(gid) = i;
end

for i = 1:Ne
    gid = idxEO(i);
    trackName{gid} = sprintf('EO-%d', i);
    sourceType(gid) = 2;
    localID(gid) = i;
end

for i = 1:Nm
    gid = idxESM(i);
    trackName{gid} = sprintf('ESM-%d', i);
    sourceType(gid) = 3;
    localID(gid) = i;
end

AssocParam.AssocMatrix = AssocMatrix;

AssocParam.Nr = Nr;
AssocParam.Ne = Ne;
AssocParam.Nm = Nm;
AssocParam.Ntotal = Ntotal;

AssocParam.idxRadar = idxRadar;
AssocParam.idxEO = idxEO;
AssocParam.idxESM = idxESM;

AssocParam.trackName = trackName;
AssocParam.sourceType = sourceType;
AssocParam.localID = localID;

AssocParam.description = ...
    '航迹关联矩阵：1表示允许关联，0表示禁止关联；默认同源航迹不能关联，无有效点航迹不能关联。';

end

%% ========================================================================
% 局部函数：安全除法
%% ========================================================================
function r = safe_div(a, b)

if b == 0
    r = NaN;
else
    r = a / b;
end

end

%% ========================================================================
% 局部函数：Radar 量测反算到世界坐标系
%% ========================================================================
function pWorld = local_radar_meas_to_world(range, az, el, pUAV, euler)

xBody = range * cos(el) * cos(az);
yBody = range * cos(el) * sin(az);
zBody = range * sin(el);

pBody = [xBody; yBody; zBody];

Rwb = local_rpy2rot(euler);

pWorld = pUAV(:) + Rwb * pBody;
pWorld = pWorld(:).';

end

%% ========================================================================
% 局部函数：RPY 转旋转矩阵，body -> world
%% ========================================================================
function R = local_rpy2rot(euler)

roll  = euler(1);
pitch = euler(2);
yaw   = euler(3);

Rx = [1 0 0;
      0 cos(roll) -sin(roll);
      0 sin(roll)  cos(roll)];

Ry = [ cos(pitch) 0 sin(pitch);
       0          1 0;
      -sin(pitch) 0 cos(pitch)];

Rz = [cos(yaw) -sin(yaw) 0;
      sin(yaw)  cos(yaw) 0;
      0         0        1];

R = Rz * Ry * Rx;

end