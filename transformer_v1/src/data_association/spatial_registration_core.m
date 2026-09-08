function [RegisteredTrack, RegistrationStats] = spatial_registration_core( ...
    CleanTrack, AlignedObs, RegParam)

t_grid = CleanTrack.t_grid(:);
K = length(t_grid);
N_true = CleanTrack.N_true;

%% ------------------------------------------------------------------------
% 1. 提取世界坐标参考航迹
%% ------------------------------------------------------------------------
Ref = struct();

Ref.pos_ref = nan(K, N_true, 3);
Ref.valid_ref = false(K, N_true);

Ref.pos_eval = nan(K, N_true, 3);
Ref.valid_eval = false(K, N_true);

for targetID = 1:N_true

    T = CleanTrack.Target(targetID);

    [pos_ref, valid_ref, pos_true] = extract_clean_track(T, K);

    Ref.pos_ref(:, targetID, :) = reshape(pos_ref, [K, 1, 3]);
    Ref.valid_ref(:, targetID) = valid_ref;

    valid_true = all(isfinite(pos_true), 2);

    if RegParam.useTruthForRMSE && any(valid_true)
        Ref.pos_eval(:, targetID, :) = reshape(pos_true, [K, 1, 3]);
        Ref.valid_eval(:, targetID) = valid_true;
    else
        Ref.pos_eval(:, targetID, :) = reshape(pos_ref, [K, 1, 3]);
        Ref.valid_eval(:, targetID) = valid_ref;
    end
end

%% ------------------------------------------------------------------------
% 2. 相对量测统一转换到世界坐标系
%% ------------------------------------------------------------------------
WorldObs = convert_measurements_to_world(AlignedObs, Ref, RegParam, K);

%% ------------------------------------------------------------------------
% 3. 在世界坐标系下估计并补偿残余空间偏差
%% ------------------------------------------------------------------------
RegisteredTrack = struct();

RegisteredTrack.t_grid = t_grid;
RegisteredTrack.N_true = N_true;
RegisteredTrack.WorldObs = WorldObs;
RegisteredTrack.RegParam = RegParam;

Target = repmat(struct(), N_true, 1);

allErrBefore = [];
allErrAfter = [];

allDispBefore = [];
allDispAfter = [];

for targetID = 1:N_true

    pos_ref = squeeze(Ref.pos_ref(:, targetID, :));
    valid_ref = Ref.valid_ref(:, targetID);

    pos_eval = squeeze(Ref.pos_eval(:, targetID, :));
    valid_eval = Ref.valid_eval(:, targetID);

    RadarBefore = extract_target_world_obs(WorldObs.Radar, targetID, K);
    EOBefore    = extract_target_world_obs(WorldObs.EO, targetID, K);
    ESMBefore   = extract_target_world_obs(WorldObs.ESM, targetID, K);

    RadarAfter = estimate_and_compensate_world_bias( ...
        RadarBefore, ...
        pos_ref, ...
        valid_ref, ...
        RegParam.alphaRadar, ...
        RegParam.maxBiasNorm);

    EOAfter = estimate_and_compensate_world_bias( ...
        EOBefore, ...
        pos_ref, ...
        valid_ref, ...
        RegParam.alphaEO, ...
        RegParam.maxBiasNorm);

    ESMAfter = estimate_and_compensate_world_bias( ...
        ESMBefore, ...
        pos_ref, ...
        valid_ref, ...
        RegParam.alphaESM, ...
        RegParam.maxBiasNorm);

    %% --------------------------------------------------------------------
    % 多传感器配准前后融合，并计算多传感器离散度
    %% --------------------------------------------------------------------
    world_xyz_before = nan(K, 3);
    world_xyz_after  = nan(K, 3);

    valid_world_before = false(K, 1);
    valid_world_after  = false(K, 1);

    dispersion_before_ts = nan(K, 1);
    dispersion_after_ts  = nan(K, 1);

    for k = 1:K

        P_before = [];
        W_before = [];

        P_after = [];
        W_after = [];

        if RadarBefore.valid(k)
            P_before(end+1, :) = RadarBefore.pos(k, :); %#ok<AGROW>
            W_before(end+1, 1) = RegParam.weightRadar; %#ok<AGROW>
        end

        if EOBefore.valid(k)
            P_before(end+1, :) = EOBefore.pos(k, :); %#ok<AGROW>
            W_before(end+1, 1) = RegParam.weightEO; %#ok<AGROW>
        end

        if ESMBefore.valid(k)
            P_before(end+1, :) = ESMBefore.pos(k, :); %#ok<AGROW>
            W_before(end+1, 1) = RegParam.weightESM; %#ok<AGROW>
        end

        if RadarAfter.valid(k)
            P_after(end+1, :) = RadarAfter.pos(k, :); %#ok<AGROW>
            W_after(end+1, 1) = RegParam.weightRadar; %#ok<AGROW>
        end

        if EOAfter.valid(k)
            P_after(end+1, :) = EOAfter.pos(k, :); %#ok<AGROW>
            W_after(end+1, 1) = RegParam.weightEO; %#ok<AGROW>
        end

        if ESMAfter.valid(k)
            P_after(end+1, :) = ESMAfter.pos(k, :); %#ok<AGROW>
            W_after(end+1, 1) = RegParam.weightESM; %#ok<AGROW>
        end

        if ~isempty(P_before)
            world_xyz_before(k, :) = weighted_average(P_before, W_before);
            valid_world_before(k) = true;
            dispersion_before_ts(k) = sensor_dispersion(P_before);
        end

        if ~isempty(P_after)
            world_xyz_after(k, :) = weighted_average(P_after, W_after);
            valid_world_after(k) = true;
            dispersion_after_ts(k) = sensor_dispersion(P_after);
        end
    end

    %% --------------------------------------------------------------------
    % 误差统计
    %% --------------------------------------------------------------------
    idx_before = valid_world_before & valid_eval & ...
        all(isfinite(world_xyz_before), 2) & all(isfinite(pos_eval), 2);

    idx_after = valid_world_after & valid_eval & ...
        all(isfinite(world_xyz_after), 2) & all(isfinite(pos_eval), 2);

    errNormBefore = nan(K, 1);
    errNormAfter = nan(K, 1);

    if any(idx_before)
        eBefore = world_xyz_before(idx_before, :) - pos_eval(idx_before, :);
        errBefore = sqrt(sum(eBefore.^2, 2));
        errNormBefore(idx_before) = errBefore;
        allErrBefore = [allErrBefore; errBefore]; %#ok<AGROW>
    else
        errBefore = [];
    end

    if any(idx_after)
        eAfter = world_xyz_after(idx_after, :) - pos_eval(idx_after, :);
        errAfter = sqrt(sum(eAfter.^2, 2));
        errNormAfter(idx_after) = errAfter;
        allErrAfter = [allErrAfter; errAfter]; %#ok<AGROW>
    else
        errAfter = [];
    end

    metricBefore = compute_error_metrics(errBefore);
    metricAfter  = compute_error_metrics(errAfter);

    rmse_improvement = metricBefore.RMSE - metricAfter.RMSE;
    improvementRate = safe_div(rmse_improvement, metricBefore.RMSE);

    dispersion_before = mean(dispersion_before_ts(isfinite(dispersion_before_ts)));
    dispersion_after  = mean(dispersion_after_ts(isfinite(dispersion_after_ts)));

    dispersion_improvement = dispersion_before - dispersion_after;
    dispersionRate = safe_div(dispersion_improvement, dispersion_before);

    allDispBefore = [allDispBefore; dispersion_before_ts(isfinite(dispersion_before_ts))]; %#ok<AGROW>
    allDispAfter  = [allDispAfter;  dispersion_after_ts(isfinite(dispersion_after_ts))]; %#ok<AGROW>

    %% --------------------------------------------------------------------
    % 明确写入世界坐标系下的 XYZ 输出
    %% --------------------------------------------------------------------
    Target(targetID).targetID = targetID;

    % 第四模块输出的世界坐标参考航迹
    Target(targetID).world_xyz_ref = pos_ref;
    Target(targetID).valid_world_ref = valid_ref;

    % 用于评价的世界坐标轨迹，优先使用真值
    Target(targetID).world_xyz_eval = pos_eval;
    Target(targetID).valid_world_eval = valid_eval;

    % 各传感器世界坐标观测及配准结果
    Target(targetID).Sensor.Radar_world = RadarAfter;
    Target(targetID).Sensor.EO_world    = EOAfter;
    Target(targetID).Sensor.ESM_world   = ESMAfter;

    % 配准前世界坐标 XYZ
    Target(targetID).world_xyz_before = world_xyz_before;
    Target(targetID).valid_world_before = valid_world_before;

    % 配准后世界坐标 XYZ
    Target(targetID).world_xyz_after = world_xyz_after;
    Target(targetID).valid_world_after = valid_world_after;

    % 为兼容旧字段，保留常用别名
    Target(targetID).pos_ref = pos_ref;
    Target(targetID).valid_ref = valid_ref;

    Target(targetID).pos_eval = pos_eval;
    Target(targetID).valid_eval = valid_eval;

    Target(targetID).pos_before_registration = world_xyz_before;
    Target(targetID).valid_before_registration = valid_world_before;

    Target(targetID).pos_after_registration = world_xyz_after;
    Target(targetID).valid_after_registration = valid_world_after;

    % 误差时间序列
    Target(targetID).errNormBefore = errNormBefore;
    Target(targetID).errNormAfter = errNormAfter;

    % 多传感器离散度时间序列
    Target(targetID).dispersion_before_ts = dispersion_before_ts;
    Target(targetID).dispersion_after_ts = dispersion_after_ts;

    % 单目标指标
    Target(targetID).rmse_before = metricBefore.RMSE;
    Target(targetID).rmse_after = metricAfter.RMSE;
    Target(targetID).rmse_improvement = rmse_improvement;
    Target(targetID).improvementRate = improvementRate;

    Target(targetID).mae_before = metricBefore.MAE;
    Target(targetID).mae_after = metricAfter.MAE;

    Target(targetID).p95_before = metricBefore.P95;
    Target(targetID).p95_after = metricAfter.P95;

    Target(targetID).max_before = metricBefore.MaxError;
    Target(targetID).max_after = metricAfter.MaxError;

    Target(targetID).dispersion_before = dispersion_before;
    Target(targetID).dispersion_after = dispersion_after;
    Target(targetID).dispersion_improvement = dispersion_improvement;
    Target(targetID).dispersionRate = dispersionRate;
end

RegisteredTrack.Target = Target;

%% ------------------------------------------------------------------------
% 总体统计
%% ------------------------------------------------------------------------
metricBeforeTotal = compute_error_metrics(allErrBefore);
metricAfterTotal  = compute_error_metrics(allErrAfter);

RegistrationStats.rmse_before_total = metricBeforeTotal.RMSE;
RegistrationStats.rmse_after_total = metricAfterTotal.RMSE;
RegistrationStats.rmse_improvement_total = ...
    RegistrationStats.rmse_before_total - RegistrationStats.rmse_after_total;
RegistrationStats.improvementRate_total = safe_div( ...
    RegistrationStats.rmse_improvement_total, ...
    RegistrationStats.rmse_before_total);

RegistrationStats.mae_before_total = metricBeforeTotal.MAE;
RegistrationStats.mae_after_total = metricAfterTotal.MAE;
RegistrationStats.mae_improvement_total = ...
    RegistrationStats.mae_before_total - RegistrationStats.mae_after_total;

RegistrationStats.p95_before_total = metricBeforeTotal.P95;
RegistrationStats.p95_after_total = metricAfterTotal.P95;
RegistrationStats.p95_improvement_total = ...
    RegistrationStats.p95_before_total - RegistrationStats.p95_after_total;

RegistrationStats.max_before_total = metricBeforeTotal.MaxError;
RegistrationStats.max_after_total = metricAfterTotal.MaxError;
RegistrationStats.max_improvement_total = ...
    RegistrationStats.max_before_total - RegistrationStats.max_after_total;

RegistrationStats.dispersion_before_total = mean(allDispBefore(isfinite(allDispBefore)));
RegistrationStats.dispersion_after_total = mean(allDispAfter(isfinite(allDispAfter)));
RegistrationStats.dispersion_improvement_total = ...
    RegistrationStats.dispersion_before_total - RegistrationStats.dispersion_after_total;
RegistrationStats.dispersionRate_total = safe_div( ...
    RegistrationStats.dispersion_improvement_total, ...
    RegistrationStats.dispersion_before_total);

RegistrationStats.coordinate = 'world';
RegistrationStats.unit = 'meter';

end

%% ========================================================================
% 世界坐标转换：相对量测 -> 世界坐标观测
% ========================================================================
function WorldObs = convert_measurements_to_world(AlignedObs, Ref, RegParam, K)

WorldObs.Radar = convert_radar_to_world(AlignedObs, RegParam, K);
WorldObs.EO    = convert_eo_to_world(AlignedObs, Ref, RegParam, K);
WorldObs.ESM   = convert_esm_to_world(AlignedObs, Ref, RegParam, K);

end

%% ========================================================================
% Radar: [range, az, el] -> world [X,Y,Z]
% ========================================================================
function RadarWorld = convert_radar_to_world(AlignedObs, RegParam, K)

Nr = get_num_track(AlignedObs.Radar);

RadarWorld.pos = nan(K, Nr, 3);
RadarWorld.valid = false(K, Nr);
RadarWorld.truthID = zeros(K, Nr);
RadarWorld.uavID = get_uav_list(AlignedObs.Radar, Nr);
RadarWorld.uavPos = nan(K, Nr, 3);

for k = 1:K
    for rid = 1:Nr

        if ~AlignedObs.Radar.valid(k, rid)
            continue;
        end

        targetID = AlignedObs.Radar.truthID(k, rid);

        if targetID <= 0
            continue;
        end

        y = squeeze(AlignedObs.Radar.Y(k, rid, :)).';

        if numel(y) < 3 || any(~isfinite(y))
            continue;
        end

        range = y(1);
        az = y(2);
        el = y(3);

        uavID = RadarWorld.uavID(rid);

        pUAV = get_uav_pos(AlignedObs, k, uavID);
        euler = get_uav_euler(AlignedObs, k, uavID);

        pWorld = radar_to_world(range, az, el, pUAV, euler);

        if RegParam.enableDemoBias
            pWorld = pWorld + RegParam.demoWorldBias.Radar;
        end

        if all(isfinite(pWorld))
            RadarWorld.pos(k, rid, :) = reshape(pWorld, [1, 1, 3]);
            RadarWorld.valid(k, rid) = true;
            RadarWorld.truthID(k, rid) = targetID;
            RadarWorld.uavPos(k, rid, :) = reshape(pUAV, [1, 1, 3]);
        end
    end
end

end

%% ========================================================================
% EO: [az, el] + 参考航迹约束 -> world [X,Y,Z]
% ========================================================================
function EOWorld = convert_eo_to_world(AlignedObs, Ref, RegParam, K)

Ne = get_num_track(AlignedObs.EO);

EOWorld.pos = nan(K, Ne, 3);
EOWorld.valid = false(K, Ne);
EOWorld.truthID = zeros(K, Ne);
EOWorld.uavID = get_uav_list(AlignedObs.EO, Ne);
EOWorld.uavPos = nan(K, Ne, 3);

for k = 1:K
    for eid = 1:Ne

        if ~AlignedObs.EO.valid(k, eid)
            continue;
        end

        targetID = AlignedObs.EO.truthID(k, eid);

        if targetID <= 0 || targetID > size(Ref.pos_ref, 2)
            continue;
        end

        if ~Ref.valid_ref(k, targetID)
            continue;
        end

        y = squeeze(AlignedObs.EO.Y(k, eid, :)).';

        if numel(y) < 2 || any(~isfinite(y))
            continue;
        end

        az = y(1);
        el = y(2);

        uavID = EOWorld.uavID(eid);

        pUAV = get_uav_pos(AlignedObs, k, uavID);
        euler = get_uav_euler(AlignedObs, k, uavID);
        pRef = squeeze(Ref.pos_ref(k, targetID, :)).';

        pWorld = eo_to_world_with_ref(az, el, pUAV, euler, pRef);

        if RegParam.enableDemoBias
            pWorld = pWorld + RegParam.demoWorldBias.EO;
        end

        if all(isfinite(pWorld))
            EOWorld.pos(k, eid, :) = reshape(pWorld, [1, 1, 3]);
            EOWorld.valid(k, eid) = true;
            EOWorld.truthID(k, eid) = targetID;
            EOWorld.uavPos(k, eid, :) = reshape(pUAV, [1, 1, 3]);
        end
    end
end

end

%% ========================================================================
% ESM: [az] + 参考航迹约束 -> world [X,Y,Z]
% ========================================================================
function ESMWorld = convert_esm_to_world(AlignedObs, Ref, RegParam, K)

Nm = get_num_track(AlignedObs.ESM);

ESMWorld.pos = nan(K, Nm, 3);
ESMWorld.valid = false(K, Nm);
ESMWorld.truthID = zeros(K, Nm);
ESMWorld.uavID = get_uav_list(AlignedObs.ESM, Nm);
ESMWorld.uavPos = nan(K, Nm, 3);

for k = 1:K
    for mid = 1:Nm

        if ~AlignedObs.ESM.valid(k, mid)
            continue;
        end

        targetID = AlignedObs.ESM.truthID(k, mid);

        if targetID <= 0 || targetID > size(Ref.pos_ref, 2)
            continue;
        end

        if ~Ref.valid_ref(k, targetID)
            continue;
        end

        y = squeeze(AlignedObs.ESM.Y(k, mid, :)).';

        if isempty(y) || any(~isfinite(y))
            continue;
        end

        az = y(1);

        uavID = ESMWorld.uavID(mid);

        pUAV = get_uav_pos(AlignedObs, k, uavID);
        euler = get_uav_euler(AlignedObs, k, uavID);
        pRef = squeeze(Ref.pos_ref(k, targetID, :)).';

        pWorld = esm_to_world_with_ref(az, pUAV, euler, pRef);

        if RegParam.enableDemoBias
            pWorld = pWorld + RegParam.demoWorldBias.ESM;
        end

        if all(isfinite(pWorld))
            ESMWorld.pos(k, mid, :) = reshape(pWorld, [1, 1, 3]);
            ESMWorld.valid(k, mid) = true;
            ESMWorld.truthID(k, mid) = targetID;
            ESMWorld.uavPos(k, mid, :) = reshape(pUAV, [1, 1, 3]);
        end
    end
end

end

%% ========================================================================
% 提取某一目标对应的世界坐标观测
% ========================================================================
function Obs = extract_target_world_obs(WorldSensor, targetID, K)

Obs.pos = nan(K, 3);
Obs.valid = false(K, 1);
Obs.bias = nan(K, 3);

for k = 1:K

    idx = find(WorldSensor.valid(k, :) & ...
        WorldSensor.truthID(k, :) == targetID);

    if isempty(idx)
        continue;
    end

    P = [];

    for ii = 1:length(idx)
        p = squeeze(WorldSensor.pos(k, idx(ii), :)).';
        if all(isfinite(p))
            P(end+1, :) = p; %#ok<AGROW>
        end
    end

    if ~isempty(P)
        Obs.pos(k, :) = median(P, 1);
        Obs.valid(k) = true;
    end
end

end

%% ========================================================================
% 世界坐标偏差估计与补偿
% ========================================================================
function ObsAfter = estimate_and_compensate_world_bias( ...
    ObsBefore, pos_ref, valid_ref, alpha, maxBiasNorm)

K = size(pos_ref, 1);

ObsAfter = ObsBefore;
ObsAfter.pos = ObsBefore.pos;
ObsAfter.valid = false(K, 1);
ObsAfter.bias = nan(K, 3);

b = [0, 0, 0];
hasInit = false;

for k = 1:K

    if ObsBefore.valid(k) && valid_ref(k) && ...
            all(isfinite(ObsBefore.pos(k, :))) && ...
            all(isfinite(pos_ref(k, :)))

        residual = ObsBefore.pos(k, :) - pos_ref(k, :);

        if ~hasInit
            b = residual;
            hasInit = true;
        else
            b = (1 - alpha) * b + alpha * residual;
        end

        b = limit_bias_norm(b, maxBiasNorm);

        ObsAfter.bias(k, :) = b;
        ObsAfter.pos(k, :) = ObsBefore.pos(k, :) - b;
        ObsAfter.valid(k) = true;

    else

        if k > 1
            ObsAfter.bias(k, :) = ObsAfter.bias(k-1, :);
        else
            ObsAfter.bias(k, :) = b;
        end

        ObsAfter.bias(k, :) = limit_bias_norm(ObsAfter.bias(k, :), maxBiasNorm);

        if ObsBefore.valid(k) && all(isfinite(ObsAfter.bias(k, :)))
            ObsAfter.pos(k, :) = ObsBefore.pos(k, :) - ObsAfter.bias(k, :);
            ObsAfter.valid(k) = true;
        end
    end
end

end

%% ========================================================================
% 限制空间补偿幅值
% ========================================================================
function b = limit_bias_norm(b, maxBiasNorm)

if nargin < 2 || isempty(maxBiasNorm) || ~isfinite(maxBiasNorm)
    return;
end

if any(~isfinite(b))
    return;
end

nb = norm(b);

if nb > maxBiasNorm && nb > 0
    b = b / nb * maxBiasNorm;
end

end

%% ========================================================================
% 构造指标表
% ========================================================================
function MetricTable = build_registration_metric_table(RegisteredTrack, RegistrationStats)

N_true = RegisteredTrack.N_true;

TargetID = zeros(N_true + 1, 1);

RMSE_before = nan(N_true + 1, 1);
RMSE_after = nan(N_true + 1, 1);
RMSE_improvement = nan(N_true + 1, 1);
RMSE_improvementRate_percent = nan(N_true + 1, 1);

MAE_before = nan(N_true + 1, 1);
MAE_after = nan(N_true + 1, 1);
MAE_improvement = nan(N_true + 1, 1);

P95_before = nan(N_true + 1, 1);
P95_after = nan(N_true + 1, 1);
P95_improvement = nan(N_true + 1, 1);

MaxError_before = nan(N_true + 1, 1);
MaxError_after = nan(N_true + 1, 1);
MaxError_improvement = nan(N_true + 1, 1);

Dispersion_before = nan(N_true + 1, 1);
Dispersion_after = nan(N_true + 1, 1);
Dispersion_improvement = nan(N_true + 1, 1);
Dispersion_improvementRate_percent = nan(N_true + 1, 1);

for i = 1:N_true

    T = RegisteredTrack.Target(i);

    TargetID(i) = i;

    RMSE_before(i) = T.rmse_before;
    RMSE_after(i) = T.rmse_after;
    RMSE_improvement(i) = T.rmse_improvement;
    RMSE_improvementRate_percent(i) = 100 * T.improvementRate;

    MAE_before(i) = T.mae_before;
    MAE_after(i) = T.mae_after;
    MAE_improvement(i) = T.mae_before - T.mae_after;

    P95_before(i) = T.p95_before;
    P95_after(i) = T.p95_after;
    P95_improvement(i) = T.p95_before - T.p95_after;

    MaxError_before(i) = T.max_before;
    MaxError_after(i) = T.max_after;
    MaxError_improvement(i) = T.max_before - T.max_after;

    Dispersion_before(i) = T.dispersion_before;
    Dispersion_after(i) = T.dispersion_after;
    Dispersion_improvement(i) = T.dispersion_improvement;
    Dispersion_improvementRate_percent(i) = 100 * T.dispersionRate;
end

% 最后一行：总体指标，TargetID = 0
idx = N_true + 1;

TargetID(idx) = 0;

RMSE_before(idx) = RegistrationStats.rmse_before_total;
RMSE_after(idx) = RegistrationStats.rmse_after_total;
RMSE_improvement(idx) = RegistrationStats.rmse_improvement_total;
RMSE_improvementRate_percent(idx) = 100 * RegistrationStats.improvementRate_total;

MAE_before(idx) = RegistrationStats.mae_before_total;
MAE_after(idx) = RegistrationStats.mae_after_total;
MAE_improvement(idx) = RegistrationStats.mae_improvement_total;

P95_before(idx) = RegistrationStats.p95_before_total;
P95_after(idx) = RegistrationStats.p95_after_total;
P95_improvement(idx) = RegistrationStats.p95_improvement_total;

MaxError_before(idx) = RegistrationStats.max_before_total;
MaxError_after(idx) = RegistrationStats.max_after_total;
MaxError_improvement(idx) = RegistrationStats.max_improvement_total;

Dispersion_before(idx) = RegistrationStats.dispersion_before_total;
Dispersion_after(idx) = RegistrationStats.dispersion_after_total;
Dispersion_improvement(idx) = RegistrationStats.dispersion_improvement_total;
Dispersion_improvementRate_percent(idx) = 100 * RegistrationStats.dispersionRate_total;

MetricTable = table( ...
    TargetID, ...
    RMSE_before, ...
    RMSE_after, ...
    RMSE_improvement, ...
    RMSE_improvementRate_percent, ...
    MAE_before, ...
    MAE_after, ...
    MAE_improvement, ...
    P95_before, ...
    P95_after, ...
    P95_improvement, ...
    MaxError_before, ...
    MaxError_after, ...
    MaxError_improvement, ...
    Dispersion_before, ...
    Dispersion_after, ...
    Dispersion_improvement, ...
    Dispersion_improvementRate_percent);

end

%% ========================================================================
% 绘制 RMSE 精度提升效果图
% ========================================================================
function plot_rmse_improvement_figure(MetricTable)

idxTarget = MetricTable.TargetID > 0;

targetID = MetricTable.TargetID(idxTarget);
rmseBefore = MetricTable.RMSE_before(idxTarget);
rmseAfter  = MetricTable.RMSE_after(idxTarget);

idxTotal = MetricTable.TargetID == 0;

if any(idxTotal)
    rmseBeforeTotal = MetricTable.RMSE_before(idxTotal);
    rmseAfterTotal  = MetricTable.RMSE_after(idxTotal);
    improveTotal = MetricTable.RMSE_improvement(idxTotal);
    improveRateTotal = MetricTable.RMSE_improvementRate_percent(idxTotal);
else
    rmseBeforeTotal = NaN;
    rmseAfterTotal = NaN;
    improveTotal = NaN;
    improveRateTotal = NaN;
end

fig = figure( ...
    'Name', '空间配准RMSE精度提升效果', ...
    'NumberTitle', 'off', ...
    'Color', 'w', ...
    'Position', [120 120 1050 520]);

hold on;
grid on;
box on;

Y = [rmseBefore(:), rmseAfter(:)];

bar(Y, 'grouped');

xlabel('目标编号');
ylabel('RMSE / m');

title(sprintf('空间配准前后RMSE对比：总体 %.2f m → %.2f m，改善 %.2f%%', ...
    rmseBeforeTotal, rmseAfterTotal, improveRateTotal), ...
    'FontWeight', 'normal');

legend('配准前 RMSE', '配准后 RMSE', 'Location', 'best');

set(gca, ...
    'XTick', 1:length(targetID), ...
    'XTickLabel', string(targetID), ...
    'FontSize', 11, ...
    'LineWidth', 1.0);

for i = 1:length(targetID)

    if isfinite(rmseBefore(i))
        text(i - 0.15, rmseBefore(i), sprintf('%.1f', rmseBefore(i)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 9);
    end

    if isfinite(rmseAfter(i))
        text(i + 0.15, rmseAfter(i), sprintf('%.1f', rmseAfter(i)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 9);
    end
end

annotationText = {
    sprintf('总体RMSE：%.2f m → %.2f m', rmseBeforeTotal, rmseAfterTotal)
    sprintf('RMSE提升量：%.2f m', improveTotal)
    sprintf('RMSE改善率：%.2f%%', improveRateTotal)
    };

annotation('textbox', [0.67, 0.68, 0.25, 0.18], ...
    'String', annotationText, ...
    'FitBoxToText', 'on', ...
    'BackgroundColor', 'w', ...
    'EdgeColor', [0.3 0.3 0.3], ...
    'FontSize', 10);

exportgraphics(fig, 'SpatialRegistration_RMSE_Improvement.png', 'Resolution', 300);

end

%% ========================================================================
% 基础工具函数
% ========================================================================
function [pos_ref, valid_ref, pos_true] = extract_clean_track(T, K)

if isfield(T, 'pos_clean')
    pos_ref = T.pos_clean;
elseif isfield(T, 'pos_completed')
    pos_ref = T.pos_completed;
elseif isfield(T, 'pos_raw')
    pos_ref = T.pos_raw;
else
    error('目标结构中找不到 pos_clean / pos_completed / pos_raw。');
end

if isfield(T, 'valid_clean')
    valid_ref = T.valid_clean;
elseif isfield(T, 'valid_completed')
    valid_ref = T.valid_completed;
elseif isfield(T, 'valid_raw')
    valid_ref = T.valid_raw;
else
    valid_ref = all(isfinite(pos_ref), 2);
end

if isfield(T, 'pos_true')
    pos_true = T.pos_true;
else
    pos_true = nan(size(pos_ref));
end

pos_ref = fix_pos_size(pos_ref, K);
pos_true = fix_pos_size(pos_true, K);

valid_ref = valid_ref(:);

if length(valid_ref) ~= K
    valid_ref = all(isfinite(pos_ref), 2);
end

valid_ref = valid_ref & all(isfinite(pos_ref), 2);

end

function P = fix_pos_size(P, K)

if isempty(P)
    P = nan(K, 3);
    return;
end

if size(P, 1) ~= K && size(P, 2) == K
    P = P.';
end

if size(P, 2) > 3
    P = P(:, 1:3);
end

if size(P, 2) < 3
    P(:, end+1:3) = NaN;
end

if size(P, 1) < K
    P(end+1:K, :) = NaN;
elseif size(P, 1) > K
    P = P(1:K, :);
end

end

function n = get_num_track(S)

if isfield(S, 'Y')
    n = size(S.Y, 2);
else
    n = 0;
end

end

function uavList = get_uav_list(S, N)

uavList = ones(1, N);

if isfield(S, 'uavID')
    temp = S.uavID(:).';
    L = min(N, length(temp));
    uavList(1:L) = temp(1:L);
end

uavList(~isfinite(uavList) | uavList < 1) = 1;

end

function pUAV = get_uav_pos(AlignedObs, k, uavID)

nUAV = size(AlignedObs.truth.uav_pos_all, 2);
uavID = min(max(round(uavID), 1), nUAV);

pUAV = squeeze(AlignedObs.truth.uav_pos_all(k, uavID, :)).';

end

function euler = get_uav_euler(AlignedObs, k, uavID)

nUAV = size(AlignedObs.truth.uav_euler_all, 2);
uavID = min(max(round(uavID), 1), nUAV);

euler = squeeze(AlignedObs.truth.uav_euler_all(k, uavID, :)).';

end

function pWorld = radar_to_world(range, az, el, pUAV, euler)

pBody = [
    range * cos(el) * cos(az);
    range * cos(el) * sin(az);
    range * sin(el)
    ];

Rwb = rpy_to_rot(euler);

pWorld = pUAV(:) + Rwb * pBody;
pWorld = pWorld(:).';

end

function pWorld = eo_to_world_with_ref(az, el, pUAV, euler, pRef)

Rwb = rpy_to_rot(euler);

dBody = [
    cos(el) * cos(az);
    cos(el) * sin(az);
    sin(el)
    ];

dWorld = Rwb * dBody;
dWorld = dWorld / max(norm(dWorld), eps);

lambda = dot(pRef(:) - pUAV(:), dWorld);

if ~isfinite(lambda) || lambda <= 0
    lambda = norm(pRef(:) - pUAV(:));
end

pWorld = pUAV(:) + lambda * dWorld;
pWorld = pWorld(:).';

end

function pWorld = esm_to_world_with_ref(az, pUAV, euler, pRef)

Rwb = rpy_to_rot(euler);

pRefBody = Rwb' * (pRef(:) - pUAV(:));

rho = norm(pRefBody(1:2));
zBody = pRefBody(3);

if ~isfinite(rho) || rho <= 0
    rho = norm(pRef(:) - pUAV(:));
end

pBody = [
    rho * cos(az);
    rho * sin(az);
    zBody
    ];

pWorld = pUAV(:) + Rwb * pBody;
pWorld = pWorld(:).';

end

function R = rpy_to_rot(euler)

roll  = euler(1);
pitch = euler(2);
yaw   = euler(3);

Rx = [
    1 0 0;
    0 cos(roll) -sin(roll);
    0 sin(roll) cos(roll)
    ];

Ry = [
    cos(pitch) 0 sin(pitch);
    0 1 0;
    -sin(pitch) 0 cos(pitch)
    ];

Rz = [
    cos(yaw) -sin(yaw) 0;
    sin(yaw) cos(yaw) 0;
    0 0 1
    ];

R = Rz * Ry * Rx;

end

function p = weighted_average(P, w)

w = w(:);

if isempty(P)
    p = [NaN, NaN, NaN];
    return;
end

if isempty(w) || length(w) ~= size(P, 1) || sum(w) <= 0
    p = mean(P, 1);
else
    p = sum(P .* w, 1) / sum(w);
end

end

function d = sensor_dispersion(P)

if isempty(P) || size(P, 1) < 2
    d = NaN;
    return;
end

center = mean(P, 1);
d = mean(vecnorm(P - center, 2, 2));

end

function metric = compute_error_metrics(err)

err = err(:);
err = err(isfinite(err));

if isempty(err)
    metric.RMSE = NaN;
    metric.MAE = NaN;
    metric.P95 = NaN;
    metric.MaxError = NaN;
    return;
end

metric.RMSE = sqrt(mean(err.^2));
metric.MAE = mean(abs(err));
metric.P95 = prctile(err, 95);
metric.MaxError = max(err);

end

function r = safe_div(a, b)

if b == 0 || ~isfinite(b)
    r = NaN;
else
    r = a / b;
end

end