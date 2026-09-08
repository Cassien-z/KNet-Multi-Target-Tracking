function [CompletedTrack, CompletionStats] = track_completion_simple(CompletionInput)
%TRACK_COMPLETION_SIMPLE 第三模块：航迹缺失数据补全核心函数
%
% 核心思想：
%   1. 基于第二模块的航迹关联结果，构建目标级三维航迹；
%   2. 根据 valid_raw 识别内部缺失段；
%   3. 根据缺失段长度采用不同补全方法：
%      - 短缺失：局部插值补全；
%      - 中等缺失：运动模型约束补全；
%      - 长缺失：局部连接 + 全局趋势约束补全；
%   4. 输出补全后的目标级航迹，并记录哪些点是补全点。
%
% 输入：
%   CompletionInput.t_grid
%   CompletionInput.AlignedObs
%   CompletionInput.AssocOut
%   CompletionInput.AssocStats
%
% 输出：
%   CompletedTrack
%   CompletionStats

%% ========================================================================
% 1. 读取输入
% ========================================================================
AlignedObs = CompletionInput.AlignedObs;
t_grid = CompletionInput.t_grid(:);

K = length(t_grid);
Nr = size(AlignedObs.Radar.Y, 2);

if isfield(AlignedObs, 'truth') && isfield(AlignedObs.truth, 'N_true')
    N_true = AlignedObs.truth.N_true;
elseif isfield(AlignedObs, 'truth') && isfield(AlignedObs.truth, 'target_pos')
    N_true = size(AlignedObs.truth.target_pos, 2);
else
    N_true = max(AlignedObs.Radar.truthID(:));
end

dt = median(diff(t_grid));

%% ========================================================================
% 2. 参数设置
% ========================================================================
param.shortGapMax = 5;       % 短缺失段：局部插值
param.midGapMax   = 25;      % 中等缺失段：运动模型补全
param.minFitPts   = 6;       % 全局趋势拟合最少点数

CompletedTrack.t_grid = t_grid;
CompletedTrack.N_true = N_true;
CompletedTrack.description = ...
    'Track completion results based on associated and time-aligned trajectories';

Target = repmat(struct(), N_true, 1);

allFilledErr = [];
allRawErr = [];
allCompletedErr = [];
allBoundaryJump = [];

totalMissingInternal = 0;
totalFilled = 0;

%% ========================================================================
% 3. 对每个目标分别进行目标级航迹重构与补全
% ========================================================================
for targetID = 1:N_true

    %% --------------------------------------------------------------------
    % 3.1 基于 Radar 量测重构目标级三维原始航迹
    %% --------------------------------------------------------------------
    pos_raw = nan(K, 3);
    count_raw = zeros(K, 1);
    sourceIDs = cell(K, 1);

    for rid = 1:Nr

        valid = AlignedObs.Radar.valid(:, rid);

        if isfield(AlignedObs.Radar, 'truthID')
            valid = valid & (AlignedObs.Radar.truthID(:, rid) == targetID);
        end

        idx = find(valid);

        for ii = 1:length(idx)

            k = idx(ii);

            rMeas = squeeze(AlignedObs.Radar.Y(k, rid, :)).';

            range = rMeas(1);
            az    = rMeas(2);
            el    = rMeas(3);

            pUAV  = AlignedObs.truth.uav_pos(k, :);
            euler = AlignedObs.truth.uav_euler(k, :);

            pWorld = local_radar_meas_to_world(range, az, el, pUAV, euler);

            if count_raw(k) == 0
                pos_raw(k, :) = pWorld;
            else
                pos_raw(k, :) = pos_raw(k, :) + pWorld;
            end

            count_raw(k) = count_raw(k) + 1;
            sourceIDs{k}{end+1} = sprintf('Radar-%d', rid); %#ok<AGROW>
        end
    end

    valid_raw = count_raw > 0;

    for k = find(valid_raw).'
        pos_raw(k, :) = pos_raw(k, :) ./ count_raw(k);
    end

    %% --------------------------------------------------------------------
    % 3.2 初始化补全结果
    %% --------------------------------------------------------------------
    pos_completed = pos_raw;
    valid_completed = valid_raw;

    filledMask = false(K, 1);

    fillMethod = repmat({''}, K, 1);
    fillMethod(valid_raw) = repmat({'raw radar'}, sum(valid_raw), 1);

    %% --------------------------------------------------------------------
    % 3.3 识别内部缺失区间
    %% --------------------------------------------------------------------
    missingSegments = local_find_missing_segments(valid_raw);

    numMissingInternal = 0;

    for sIdx = 1:length(missingSegments)
        numMissingInternal = numMissingInternal + missingSegments(sIdx).length;
    end

    totalMissingInternal = totalMissingInternal + numMissingInternal;

    %% --------------------------------------------------------------------
    % 3.4 分级补全
    %% --------------------------------------------------------------------
    for sIdx = 1:length(missingSegments)

        seg = missingSegments(sIdx);

        gapIdx = seg.startIdx:seg.endIdx;
        gapLen = seg.length;

        leftIdx = seg.startIdx - 1;
        rightIdx = seg.endIdx + 1;

        if leftIdx < 1 || rightIdx > K
            continue;
        end

        if ~valid_completed(leftIdx) || ~valid_completed(rightIdx)
            continue;
        end

        tGap = t_grid(gapIdx);

        t0 = t_grid(leftIdx);
        t1 = t_grid(rightIdx);

        p0 = pos_completed(leftIdx, :);
        p1 = pos_completed(rightIdx, :);

        if gapLen <= param.shortGapMax

            methodName = 'local interpolation';

            validIdx = find(valid_raw);

            if length(validIdx) >= 4
                fillPos = interp1(t_grid(validIdx), pos_raw(validIdx, :), ...
                    tGap, 'pchip', 'extrap');
            else
                fillPos = interp1([t0; t1], [p0; p1], tGap, 'linear');
            end

        elseif gapLen <= param.midGapMax

            methodName = 'motion model';

            v0 = local_estimate_velocity(pos_completed, valid_completed, leftIdx, dt);
            v1 = local_estimate_velocity(pos_completed, valid_completed, rightIdx, dt);

            fillPos = local_hermite_fill(t0, p0, v0, t1, p1, v1, tGap);

        else

            methodName = 'trend constrained';

            linearPos = interp1([t0; t1], [p0; p1], tGap, 'linear');

            validIdx = find(valid_raw);

            if length(validIdx) >= param.minFitPts

                trendPos = zeros(length(tGap), 3);

                tauAll = (t_grid(validIdx) - t_grid(1)) / ...
                    max(t_grid(end) - t_grid(1), eps);

                tauGap = (tGap - t_grid(1)) / ...
                    max(t_grid(end) - t_grid(1), eps);

                for d = 1:3
                    pp = polyfit(tauAll, pos_raw(validIdx, d), 2);
                    trendPos(:, d) = polyval(pp, tauGap);
                end

                fillPos = 0.70 * linearPos + 0.30 * trendPos;

            else
                fillPos = linearPos;
            end
        end

        pos_completed(gapIdx, :) = fillPos;
        valid_completed(gapIdx) = true;
        filledMask(gapIdx) = true;
        fillMethod(gapIdx) = repmat({methodName}, length(gapIdx), 1);

        missingSegments(sIdx).fillMethod = methodName;
        missingSegments(sIdx).filled = true;

        jumpValue = local_boundary_velocity_jump(pos_completed, leftIdx, rightIdx, dt);

        missingSegments(sIdx).boundaryVelocityJump = jumpValue;
        allBoundaryJump = [allBoundaryJump; jumpValue]; %#ok<AGROW>
    end

    totalFilled = totalFilled + sum(filledMask);

    %% --------------------------------------------------------------------
    % 3.5 误差计算
    %% --------------------------------------------------------------------
    pos_true = nan(K, 3);
    err_raw = nan(K, 1);
    err_completed = nan(K, 1);

    if isfield(AlignedObs, 'truth') && isfield(AlignedObs.truth, 'target_pos')

        pos_true = squeeze(AlignedObs.truth.target_pos(:, targetID, :));

        err_raw(valid_raw) = sqrt(sum( ...
            (pos_raw(valid_raw, :) - pos_true(valid_raw, :)).^2, 2));

        err_completed(valid_completed) = sqrt(sum( ...
            (pos_completed(valid_completed, :) - pos_true(valid_completed, :)).^2, 2));

        allRawErr = [allRawErr; err_raw(valid_raw)]; %#ok<AGROW>
        allCompletedErr = [allCompletedErr; err_completed(valid_completed)]; %#ok<AGROW>
        allFilledErr = [allFilledErr; err_completed(filledMask)]; %#ok<AGROW>
    end

    %% --------------------------------------------------------------------
    % 3.6 当前目标统计指标
    %% --------------------------------------------------------------------
    numRaw = sum(valid_raw);
    numFilled = sum(filledMask);
    numCompleted = sum(valid_completed);

    fillCoverage = numFilled / max(numMissingInternal, 1);

    if any(filledMask)
        rmse_filled = sqrt(mean(err_completed(filledMask).^2, 'omitnan'));
        mean_filled_error = mean(err_completed(filledMask), 'omitnan');
    else
        rmse_filled = NaN;
        mean_filled_error = NaN;
    end

    if any(valid_raw)
        rmse_raw = sqrt(mean(err_raw(valid_raw).^2, 'omitnan'));
        mean_raw_error = mean(err_raw(valid_raw), 'omitnan');
    else
        rmse_raw = NaN;
        mean_raw_error = NaN;
    end

    if any(valid_completed)
        rmse_completed = sqrt(mean(err_completed(valid_completed).^2, 'omitnan'));
        mean_completed_error = mean(err_completed(valid_completed), 'omitnan');
    else
        rmse_completed = NaN;
        mean_completed_error = NaN;
    end

    boundaryJumpList = [];

    for sIdx = 1:length(missingSegments)
        if isfield(missingSegments(sIdx), 'boundaryVelocityJump') && ...
                ~isempty(missingSegments(sIdx).boundaryVelocityJump)
            boundaryJumpList = [boundaryJumpList; ...
                missingSegments(sIdx).boundaryVelocityJump]; %#ok<AGROW>
        end
    end

    if ~isempty(boundaryJumpList)
        mean_boundary_velocity_jump = mean(boundaryJumpList, 'omitnan');
    else
        mean_boundary_velocity_jump = NaN;
    end

    %% --------------------------------------------------------------------
    % 3.7 保存当前目标结果
    %% --------------------------------------------------------------------
    Target(targetID).targetID = targetID;

    Target(targetID).pos_raw = pos_raw;
    Target(targetID).valid_raw = valid_raw;

    Target(targetID).pos_completed = pos_completed;
    Target(targetID).valid_completed = valid_completed;

    Target(targetID).filledMask = filledMask;
    Target(targetID).fillMethod = fillMethod;
    Target(targetID).missingSegments = missingSegments;

    Target(targetID).pos_true = pos_true;

    Target(targetID).err_raw = err_raw;
    Target(targetID).err_completed = err_completed;

    Target(targetID).numRaw = numRaw;
    Target(targetID).numMissingInternal = numMissingInternal;
    Target(targetID).numFilled = numFilled;
    Target(targetID).numCompleted = numCompleted;

    Target(targetID).fillCoverage = fillCoverage;

    Target(targetID).rmse_raw = rmse_raw;
    Target(targetID).mean_raw_error = mean_raw_error;

    Target(targetID).rmse_filled = rmse_filled;
    Target(targetID).mean_filled_error = mean_filled_error;

    Target(targetID).rmse_completed = rmse_completed;
    Target(targetID).mean_completed_error = mean_completed_error;

    Target(targetID).mean_boundary_velocity_jump = mean_boundary_velocity_jump;

    Target(targetID).sourceIDs = sourceIDs;
end

CompletedTrack.Target = Target;

%% ========================================================================
% 4. 总体统计
% ========================================================================
CompletionStats.N_true = N_true;
CompletionStats.K = K;

CompletionStats.totalRaw = 0;
CompletionStats.totalMissingInternal = 0;
CompletionStats.totalFilled = 0;
CompletionStats.totalCompleted = 0;

for targetID = 1:N_true
    CompletionStats.totalRaw = CompletionStats.totalRaw + Target(targetID).numRaw;
    CompletionStats.totalMissingInternal = CompletionStats.totalMissingInternal + ...
        Target(targetID).numMissingInternal;
    CompletionStats.totalFilled = CompletionStats.totalFilled + Target(targetID).numFilled;
    CompletionStats.totalCompleted = CompletionStats.totalCompleted + Target(targetID).numCompleted;
end

CompletionStats.fillCoverage = ...
    CompletionStats.totalFilled / max(CompletionStats.totalMissingInternal, 1);

if ~isempty(allFilledErr)
    CompletionStats.rmse_filled = sqrt(mean(allFilledErr.^2, 'omitnan'));
    CompletionStats.mean_filled_error = mean(allFilledErr, 'omitnan');
else
    CompletionStats.rmse_filled = NaN;
    CompletionStats.mean_filled_error = NaN;
end

if ~isempty(allRawErr)
    CompletionStats.rmse_raw = sqrt(mean(allRawErr.^2, 'omitnan'));
    CompletionStats.mean_raw_error = mean(allRawErr, 'omitnan');
else
    CompletionStats.rmse_raw = NaN;
    CompletionStats.mean_raw_error = NaN;
end

if ~isempty(allCompletedErr)
    CompletionStats.rmse_completed = sqrt(mean(allCompletedErr.^2, 'omitnan'));
    CompletionStats.mean_completed_error = mean(allCompletedErr, 'omitnan');
else
    CompletionStats.rmse_completed = NaN;
    CompletionStats.mean_completed_error = NaN;
end

if ~isempty(allBoundaryJump)
    CompletionStats.mean_boundary_velocity_jump = mean(allBoundaryJump, 'omitnan');
    CompletionStats.max_boundary_velocity_jump = max(allBoundaryJump);
else
    CompletionStats.mean_boundary_velocity_jump = NaN;
    CompletionStats.max_boundary_velocity_jump = NaN;
end

CompletedTrack.Stats = CompletionStats;

end

%% ========================================================================
% 局部函数：寻找内部缺失段
% ========================================================================
function missingSegments = local_find_missing_segments(valid)

valid = valid(:);

missingSegments = struct( ...
    'startIdx', {}, ...
    'endIdx', {}, ...
    'length', {}, ...
    'type', {}, ...
    'fillMethod', {}, ...
    'filled', {}, ...
    'boundaryVelocityJump', {});

if ~any(valid)
    return;
end

firstValid = find(valid, 1, 'first');
lastValid  = find(valid, 1, 'last');

missing = ~valid;

missing(1:firstValid-1) = false;
missing(lastValid+1:end) = false;

d = diff([false; missing; false]);

startList = find(d == 1);
endList = find(d == -1) - 1;

for i = 1:length(startList)

    L = endList(i) - startList(i) + 1;

    if L <= 5
        typeName = 'short';
    elseif L <= 25
        typeName = 'middle';
    else
        typeName = 'long';
    end

    missingSegments(end+1).startIdx = startList(i); %#ok<AGROW>
    missingSegments(end).endIdx = endList(i);
    missingSegments(end).length = L;
    missingSegments(end).type = typeName;
    missingSegments(end).fillMethod = '';
    missingSegments(end).filled = false;
    missingSegments(end).boundaryVelocityJump = NaN;
end

end

%% ========================================================================
% 局部函数：估计速度
% ========================================================================
function v = local_estimate_velocity(pos, valid, idx, dt)

idxPrev = find(valid(1:idx-1), 1, 'last');
idxNextRel = find(valid(idx+1:end), 1, 'first');

if ~isempty(idxNextRel)
    idxNext = idx + idxNextRel;
else
    idxNext = [];
end

if ~isempty(idxPrev) && ~isempty(idxNext)
    v = (pos(idxNext, :) - pos(idxPrev, :)) / ((idxNext - idxPrev) * dt);
elseif ~isempty(idxPrev)
    v = (pos(idx, :) - pos(idxPrev, :)) / ((idx - idxPrev) * dt);
elseif ~isempty(idxNext)
    v = (pos(idxNext, :) - pos(idx, :)) / ((idxNext - idx) * dt);
else
    v = zeros(1, 3);
end

end

%% ========================================================================
% 局部函数：三次 Hermite 补全
% ========================================================================
function fillPos = local_hermite_fill(t0, p0, v0, t1, p1, v1, tGap)

T = t1 - t0;
s = (tGap - t0) / max(T, eps);

s = s(:);

h00 =  2*s.^3 - 3*s.^2 + 1;
h10 =      s.^3 - 2*s.^2 + s;
h01 = -2*s.^3 + 3*s.^2;
h11 =      s.^3 -   s.^2;

fillPos = h00 .* p0 + h10 .* (T*v0) + h01 .* p1 + h11 .* (T*v1);

end

%% ========================================================================
% 局部函数：计算缺失段边界速度跳变
% ========================================================================
function jumpValue = local_boundary_velocity_jump(pos_completed, leftIdx, rightIdx, dt)

if rightIdx <= leftIdx + 1
    jumpValue = NaN;
    return;
end

firstFillIdx = leftIdx + 1;
lastFillIdx  = rightIdx - 1;

vLeft = (pos_completed(firstFillIdx, :) - pos_completed(leftIdx, :)) / dt;
vRight = (pos_completed(rightIdx, :) - pos_completed(lastFillIdx, :)) / dt;

jumpValue = norm(vRight - vLeft, 2);

end

%% ========================================================================
% 局部函数：Radar 量测反算到世界坐标系
% ========================================================================
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
% 局部函数：RPY 转旋转矩阵
% ========================================================================
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