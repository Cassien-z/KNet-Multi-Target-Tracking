function [CleanTrack, OutlierStats] = outlier_removal_simple(CompletedTrack)
%OUTLIER_REMOVAL_SIMPLE 第四模块：航迹野值剔除核心函数
%
% 核心思想：
%   1. 以第三模块补全后的目标级航迹为输入；
%   2. 对每个目标分别进行处理；
%   3. 利用局部窗口拟合预测当前点；
%   4. 计算当前点与局部预测值之间的残差；
%   5. 使用 MAD 自适应阈值判断 X/Y/Z 各维度是否为野值；
%   6. 只修正异常维度，正常维度保留原始值；
%   7. 对连续异常点设置保护机制，避免把真实机动误判为野值。
%
% 输入：
%   CompletedTrack.Target(i).pos_completed
%   CompletedTrack.Target(i).valid_completed
%   CompletedTrack.Target(i).filledMask
%
% 输出：
%   CleanTrack
%   OutlierStats

%% ========================================================================
% 1. 读取输入
% ========================================================================
t_grid = CompletedTrack.t_grid(:);
K = length(t_grid);
N_true = CompletedTrack.N_true;

dt = median(diff(t_grid));

%% ========================================================================
% 2. 参数设置
% ========================================================================
param.winMin = 3;                  % 初始半窗口长度
param.winMax = 8;                  % 最大半窗口长度
param.minFitPts = 5;               % 局部拟合所需最少点数

param.madScale = 1.4826;           % MAD 到标准差的等效系数
param.thFactor = 4.5;              % 自适应阈值系数
param.minThreshold = 30;           % 单维最小阈值，单位 m
param.maxThreshold = 800;          % 单维最大阈值，单位 m

param.speedFactor = 3.5;           % 速度突变阈值系数
param.continuousProtectLen = 3;    % 连续异常保护长度

param.excludeFilledFromStats = false;
% false：补全点也参与野值判断；
% true ：补全点不参与阈值统计，只参与检测。

%% ========================================================================
% 3. 初始化输出
% ========================================================================
CleanTrack.t_grid = t_grid;
CleanTrack.N_true = N_true;
CleanTrack.description = ...
    'Outlier removal results based on completed target-level trajectories';

Target = repmat(struct(), N_true, 1);

totalValid = 0;
totalOutlierPoint = 0;
totalOutlierDim = 0;
totalRepairedDim = 0;

allCleanErr = [];
allBeforeErr = [];

%% ========================================================================
% 4. 对每个目标分别进行野值检测与修正
% ========================================================================
for targetID = 1:N_true

    T = CompletedTrack.Target(targetID);

    pos_before = T.pos_completed;
    valid = T.valid_completed(:);
    filledMask = T.filledMask(:);

    pos_clean = pos_before;

    outlierMask = false(K, 1);
    outlierDimMask = false(K, 3);
    repairedDimMask = false(K, 3);

    predPos = nan(K, 3);
    residual = nan(K, 3);
    threshold = nan(K, 3);

    repairMethod = repmat({''}, K, 1);
    repairMethod(valid) = repmat({'kept'}, sum(valid), 1);

    validIdxAll = find(valid);

    %% --------------------------------------------------------------------
    % 4.1 逐点检测
    %% --------------------------------------------------------------------
    for idxNow = validIdxAll(:).'

        % 边界附近可用点不足时跳过
        if idxNow <= 2 || idxNow >= K-1
            continue;
        end

        pNow = pos_clean(idxNow, :);

        if any(isnan(pNow))
            continue;
        end

        [pPred, dimTh, isReliable] = local_predict_by_adaptive_window( ...
            pos_clean, valid, filledMask, idxNow, t_grid, param);

        if ~isReliable
            continue;
        end

        predPos(idxNow, :) = pPred;
        residual(idxNow, :) = pNow - pPred;
        threshold(idxNow, :) = dimTh;

        dimOutlier = abs(residual(idxNow, :)) > dimTh;

        % 辅助运动约束：速度突变检测
        dimOutlier = dimOutlier | local_motion_jump_check( ...
            pos_clean, valid, idxNow, dt, param);

        if any(dimOutlier)
            outlierDimMask(idxNow, :) = dimOutlier;
            outlierMask(idxNow) = true;
        end
    end

    %% --------------------------------------------------------------------
    % 4.2 连续异常保护
    %
    % 如果连续多个点都被判为异常，更可能是目标发生机动，
    % 不直接全部替换，避免把真实运动变化当成野值。
    %% --------------------------------------------------------------------
    outlierMaskProtected = local_continuous_outlier_protection( ...
        outlierMask, param.continuousProtectLen);

    removedByProtection = outlierMask & ~outlierMaskProtected;

    outlierMask = outlierMaskProtected;
    outlierDimMask(removedByProtection, :) = false;

    %% --------------------------------------------------------------------
    % 4.3 分维度修正
    %% --------------------------------------------------------------------
    for k = find(outlierMask).'

        if any(isnan(predPos(k, :)))
            continue;
        end

        dimBad = outlierDimMask(k, :);

        pos_clean(k, dimBad) = predPos(k, dimBad);

        repairedDimMask(k, dimBad) = true;
        repairMethod{k} = 'dimension-wise prediction replacement';
    end

    %% --------------------------------------------------------------------
    % 4.4 修正后轻度平滑，仅作用于被修正点附近，避免航迹断裂
    %% --------------------------------------------------------------------
    pos_clean = local_light_smooth_repaired_points( ...
        pos_clean, valid, repairedDimMask);

    %% --------------------------------------------------------------------
    % 4.5 误差统计
    %% --------------------------------------------------------------------
    pos_true = nan(K, 3);
    err_before = nan(K, 1);
    err_clean = nan(K, 1);

    if isfield(T, 'pos_true') && ~all(isnan(T.pos_true(:)))

        pos_true = T.pos_true;

        err_before(valid) = sqrt(sum( ...
            (pos_before(valid, :) - pos_true(valid, :)).^2, 2));

        err_clean(valid) = sqrt(sum( ...
            (pos_clean(valid, :) - pos_true(valid, :)).^2, 2));

        allBeforeErr = [allBeforeErr; err_before(valid)]; %#ok<AGROW>
        allCleanErr = [allCleanErr; err_clean(valid)]; %#ok<AGROW>
    end

    %% --------------------------------------------------------------------
    % 4.6 当前目标统计
    %% --------------------------------------------------------------------
    numValid = sum(valid);
    numOutlierPoint = sum(outlierMask);
    numOutlierDim = sum(outlierDimMask(:));
    numRepairedDim = sum(repairedDimMask(:));

    totalValid = totalValid + numValid;
    totalOutlierPoint = totalOutlierPoint + numOutlierPoint;
    totalOutlierDim = totalOutlierDim + numOutlierDim;
    totalRepairedDim = totalRepairedDim + numRepairedDim;

    if any(valid)
        rmse_before = sqrt(mean(err_before(valid).^2, 'omitnan'));
        rmse_clean = sqrt(mean(err_clean(valid).^2, 'omitnan'));

        mae_before = mean(err_before(valid), 'omitnan');
        mae_clean = mean(err_clean(valid), 'omitnan');
    else
        rmse_before = NaN;
        rmse_clean = NaN;
        mae_before = NaN;
        mae_clean = NaN;
    end

    if ~isnan(rmse_before) && rmse_before > 0 && ~isnan(rmse_clean)
        rmse_improvement_rate = ...
            (rmse_before - rmse_clean) / rmse_before * 100;
    else
        rmse_improvement_rate = NaN;
    end

    if ~isnan(mae_before) && mae_before > 0 && ~isnan(mae_clean)
        mae_improvement_rate = ...
            (mae_before - mae_clean) / mae_before * 100;
    else
        mae_improvement_rate = NaN;
    end

    %% --------------------------------------------------------------------
    % 4.7 保存当前目标
    %% --------------------------------------------------------------------
    Target(targetID).targetID = targetID;

    Target(targetID).pos_before = pos_before;
    Target(targetID).pos_clean = pos_clean;

    Target(targetID).valid_clean = valid;
    Target(targetID).filledMask = filledMask;

    Target(targetID).outlierMask = outlierMask;
    Target(targetID).outlierDimMask = outlierDimMask;
    Target(targetID).repairedDimMask = repairedDimMask;

    Target(targetID).predPos = predPos;
    Target(targetID).residual = residual;
    Target(targetID).threshold = threshold;
    Target(targetID).repairMethod = repairMethod;

    Target(targetID).pos_true = pos_true;
    Target(targetID).err_before = err_before;
    Target(targetID).err_clean = err_clean;

    Target(targetID).numValid = numValid;
    Target(targetID).numOutlierPoint = numOutlierPoint;
    Target(targetID).numOutlierDim = numOutlierDim;
    Target(targetID).numRepairedDim = numRepairedDim;

    Target(targetID).outlierPointRate = numOutlierPoint / max(numValid, 1);
    Target(targetID).outlierDimRate = numOutlierDim / max(3*numValid, 1);

    Target(targetID).rmse_before = rmse_before;
    Target(targetID).rmse_clean = rmse_clean;
    Target(targetID).rmse_improvement = rmse_before - rmse_clean;
    Target(targetID).rmse_improvement_rate = rmse_improvement_rate;

    Target(targetID).mae_before = mae_before;
    Target(targetID).mae_clean = mae_clean;
    Target(targetID).mae_improvement = mae_before - mae_clean;
    Target(targetID).mae_improvement_rate = mae_improvement_rate;
end

CleanTrack.Target = Target;

%% ========================================================================
% 5. 总体统计
% ========================================================================
OutlierStats.N_true = N_true;
OutlierStats.K = K;

OutlierStats.totalValid = totalValid;
OutlierStats.totalOutlierPoint = totalOutlierPoint;
OutlierStats.totalOutlierDim = totalOutlierDim;
OutlierStats.totalRepairedDim = totalRepairedDim;

OutlierStats.outlierPointRate = totalOutlierPoint / max(totalValid, 1);
OutlierStats.outlierDimRate = totalOutlierDim / max(3*totalValid, 1);

if ~isempty(allBeforeErr)
    OutlierStats.rmse_before = sqrt(mean(allBeforeErr.^2, 'omitnan'));
    OutlierStats.mae_before = mean(allBeforeErr, 'omitnan');
else
    OutlierStats.rmse_before = NaN;
    OutlierStats.mae_before = NaN;
end

if ~isempty(allCleanErr)
    OutlierStats.rmse_clean = sqrt(mean(allCleanErr.^2, 'omitnan'));
    OutlierStats.mae_clean = mean(allCleanErr, 'omitnan');
else
    OutlierStats.rmse_clean = NaN;
    OutlierStats.mae_clean = NaN;
end

OutlierStats.rmse_improvement = ...
    OutlierStats.rmse_before - OutlierStats.rmse_clean;

OutlierStats.mae_improvement = ...
    OutlierStats.mae_before - OutlierStats.mae_clean;

if ~isnan(OutlierStats.rmse_before) && OutlierStats.rmse_before > 0 && ...
        ~isnan(OutlierStats.rmse_clean)

    OutlierStats.rmse_improvement_rate = ...
        (OutlierStats.rmse_before - OutlierStats.rmse_clean) / ...
        OutlierStats.rmse_before * 100;
else
    OutlierStats.rmse_improvement_rate = NaN;
end

if ~isnan(OutlierStats.mae_before) && OutlierStats.mae_before > 0 && ...
        ~isnan(OutlierStats.mae_clean)

    OutlierStats.mae_improvement_rate = ...
        (OutlierStats.mae_before - OutlierStats.mae_clean) / ...
        OutlierStats.mae_before * 100;
else
    OutlierStats.mae_improvement_rate = NaN;
end

CleanTrack.Stats = OutlierStats;

end

%% ========================================================================
% 局部函数：自适应窗口局部预测
%% ========================================================================
function [pPred, dimTh, isReliable] = local_predict_by_adaptive_window( ...
    pos, valid, filledMask, idxNow, t_grid, param)

K = size(pos, 1);

pPred = nan(1, 3);
dimTh = nan(1, 3);
isReliable = false;

for halfWin = param.winMin:param.winMax

    idxL = max(1, idxNow - halfWin);
    idxR = min(K, idxNow + halfWin);

    idxWin = idxL:idxR;
    idxWin(idxWin == idxNow) = [];

    useIdx = idxWin(valid(idxWin));

    if param.excludeFilledFromStats
        useIdx = useIdx(~filledMask(useIdx));
    end

    if length(useIdx) < param.minFitPts
        continue;
    end

    tUse = t_grid(useIdx);
    tauUse = tUse - t_grid(idxNow);
    tauNow = 0;

    pFit = nan(1, 3);
    resUse = nan(length(useIdx), 3);

    for d = 1:3

        yUse = pos(useIdx, d);

        if any(isnan(yUse))
            continue;
        end

        % 点数足够时用二次拟合，否则用线性拟合
        if length(useIdx) >= 6
            pp = polyfit(tauUse, yUse, 2);
        else
            pp = polyfit(tauUse, yUse, 1);
        end

        pFit(d) = polyval(pp, tauNow);

        yPredUse = polyval(pp, tauUse);
        resUse(:, d) = yUse - yPredUse;
    end

    if any(isnan(pFit))
        continue;
    end

    for d = 1:3

        rd = resUse(:, d);
        rd = rd(~isnan(rd));

        if isempty(rd)
            dimTh(d) = param.maxThreshold;
            continue;
        end

        medRes = median(rd);
        madRes = median(abs(rd - medRes));

        th = param.thFactor * param.madScale * madRes;

        th = max(th, param.minThreshold);
        th = min(th, param.maxThreshold);

        dimTh(d) = th;
    end

    pPred = pFit;
    isReliable = true;
    return;
end

end

%% ========================================================================
% 局部函数：速度突变辅助检测
%% ========================================================================
function dimOut = local_motion_jump_check(pos, valid, idxNow, dt, param)

dimOut = false(1, 3);

idxPrev = find(valid(1:idxNow-1), 1, 'last');
idxNextRel = find(valid(idxNow+1:end), 1, 'first');

if isempty(idxPrev) || isempty(idxNextRel)
    return;
end

idxNext = idxNow + idxNextRel;

pPrev = pos(idxPrev, :);
pNow  = pos(idxNow, :);
pNext = pos(idxNext, :);

dtPrev = (idxNow - idxPrev) * dt;
dtNext = (idxNext - idxNow) * dt;

vPrev = (pNow - pPrev) / max(dtPrev, eps);
vNext = (pNext - pNow) / max(dtNext, eps);

vJump = abs(vNext - vPrev);

idxLocal = max(1, idxNow-6):min(size(pos,1), idxNow+6);
idxLocal = idxLocal(valid(idxLocal));

if length(idxLocal) < 4
    return;
end

vList = [];

for i = 2:length(idxLocal)

    k1 = idxLocal(i-1);
    k2 = idxLocal(i);

    if k2 > k1
        vList(end+1, :) = abs((pos(k2,:) - pos(k1,:)) / ((k2-k1)*dt)); %#ok<AGROW>
    end
end

if isempty(vList)
    return;
end

for d = 1:3

    vd = vList(:, d);

    medV = median(vd);
    madV = median(abs(vd - medV));

    vTh = medV + param.speedFactor * param.madScale * madV;
    vTh = max(vTh, param.minThreshold / max(dt, eps));

    if vJump(d) > vTh
        dimOut(d) = true;
    end
end

end

%% ========================================================================
% 局部函数：连续异常保护
%% ========================================================================
function outlierMaskProtected = local_continuous_outlier_protection( ...
    outlierMask, continuousProtectLen)

outlierMask = outlierMask(:);
outlierMaskProtected = outlierMask;

if ~any(outlierMask)
    return;
end

d = diff([false; outlierMask; false]);

startList = find(d == 1);
endList = find(d == -1) - 1;

for i = 1:length(startList)

    L = endList(i) - startList(i) + 1;

    if L >= continuousProtectLen
        outlierMaskProtected(startList(i):endList(i)) = false;
    end
end

end

%% ========================================================================
% 局部函数：对修正点附近进行轻度平滑
%% ========================================================================
function posOut = local_light_smooth_repaired_points(posIn, valid, repairedDimMask)

posOut = posIn;
K = size(posIn, 1);

idxRepair = find(any(repairedDimMask, 2));

for k = idxRepair(:).'

    if k <= 1 || k >= K
        continue;
    end

    idxPrev = find(valid(1:k-1), 1, 'last');
    idxNextRel = find(valid(k+1:end), 1, 'first');

    if isempty(idxPrev) || isempty(idxNextRel)
        continue;
    end

    idxNext = k + idxNextRel;

    for d = 1:3

        if repairedDimMask(k, d)

            posOut(k, d) = ...
                0.50 * posIn(k, d) + ...
                0.25 * posIn(idxPrev, d) + ...
                0.25 * posIn(idxNext, d);
        end
    end
end

end