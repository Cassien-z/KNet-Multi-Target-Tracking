function AssocOut = track_association_simple(AlignedObs, AssocParam)
%TRACK_ASSOCIATION_SIMPLE 多源航迹关联核心函数
%
% 功能：
%   读取第一模块“时间尺度对齐”的输出 AlignedObs，
%   并结合航迹关联约束矩阵 AssocMatrix，
%   对 Radar / EO / ESM 的同步量测进行航迹关联。
%
% 输入：
%   AlignedObs.Radar.Y        K × Nr × 3, [range, azimuth, elevation]
%   AlignedObs.Radar.valid    K × Nr
%
%   AlignedObs.EO.Y           K × Ne × 2, [azimuth, elevation]
%   AlignedObs.EO.valid       K × Ne
%
%   AlignedObs.ESM.Y          K × Nm × 1, [azimuth]
%   AlignedObs.ESM.valid      K × Nm
%
%   AssocParam.AssocMatrix    Ntotal × Ntotal 逻辑矩阵
%                             1：允许关联
%                             0：禁止关联
%
% 全局编号规则：
%   Radar: 1 ~ Nr
%   EO:    Nr+1 ~ Nr+Ne
%   ESM:   Nr+Ne+1 ~ Nr+Ne+Nm
%
% 说明：
%   1. truthID 不参与关联判决，只随结果保存，供主程序统计性能使用；
%   2. 当前以 Radar 为主源，分别关联 EO 和 ESM；
%   3. 航迹关联矩阵作为硬约束，在代价计算前生效；
%   4. Radar-ESM 关联采用跨平台几何投影，不直接比较不同平台下的方位角。

%% ========================================================================
% 0. 基本尺寸
% ========================================================================
t_grid = AlignedObs.t_grid(:);
K = length(t_grid);

Nr = size(AlignedObs.Radar.Y, 2);
Ne = size(AlignedObs.EO.Y, 2);
Nm = size(AlignedObs.ESM.Y, 2);

%% ========================================================================
% 1. 参数设置
% ========================================================================
gateEO  = 0.024;
gateESM = 0.075;

totalGateEO  = 0.038;
totalGateESM = 0.095;

gateTopoEO  = 0.060;
gateTopoESM = 0.120;

lambdaTopoEO  = 0.22;
lambdaTopoESM = 0.08;

switchPenaltyEO  = 0.003;
switchPenaltyESM = 0.004;

ambDiffEO  = 0.0015;
ambDiffESM = 0.0010;

topoK = 3;

%% ========================================================================
% 2. 若未输入 AssocParam，则构造默认关联矩阵
% ========================================================================
if nargin < 2 || isempty(AssocParam)
    AssocParam = build_default_assoc_param(Nr, Ne, Nm);
end

if ~isfield(AssocParam, 'AssocMatrix')
    AssocParam = build_default_assoc_param(Nr, Ne, Nm);
end

AssocMatrix = logical(AssocParam.AssocMatrix);

Ntotal = Nr + Ne + Nm;

if size(AssocMatrix, 1) ~= Ntotal || size(AssocMatrix, 2) ~= Ntotal
    error('AssocMatrix 尺寸错误，应为 %d × %d。', Ntotal, Ntotal);
end

AssocMatrix = AssocMatrix & AssocMatrix.';
AssocMatrix(1:Ntotal+1:end) = false;

AssocParam.AssocMatrix = AssocMatrix;
AssocParam.Nr = Nr;
AssocParam.Ne = Ne;
AssocParam.Nm = Nm;
AssocParam.Ntotal = Ntotal;

%% ========================================================================
% 3. 初始化输出
% ========================================================================
AssocOut.t_grid = t_grid;
AssocOut.Frame = repmat(struct(), K, 1);
AssocOut.description = ...
    'Multi-source track association results with association constraint matrix and cross-platform ESM projection';

AssocOut.AssocParam = AssocParam;

prevEOForRadar  = nan(1, Nr);
prevESMForRadar = nan(1, Nr);

%% ========================================================================
% 4. 逐时刻关联
% ========================================================================
for k = 1:K

    tk = t_grid(k);

    radarIdx = find(AlignedObs.Radar.valid(k, :));
    eoIdx    = find(AlignedObs.EO.valid(k, :));
    esmIdx   = find(AlignedObs.ESM.valid(k, :));

    tracks = init_empty_tracks();

    if isempty(radarIdx)
        AssocOut.Frame(k).time = tk;
        AssocOut.Frame(k).tracks = tracks;
        AssocOut.Frame(k).numTracks = 0;
        continue;
    end

    %% --------------------------------------------------------------------
    % 4.1 Radar-EO 关联代价矩阵
    %% --------------------------------------------------------------------
    C_RE = inf(length(radarIdx), length(eoIdx));

    for i = 1:length(radarIdx)

        rid = radarIdx(i);
        rMeas = squeeze(AlignedObs.Radar.Y(k, rid, :)).';

        rAz = rMeas(2);
        rEl = rMeas(3);

        for j = 1:length(eoIdx)

            eid = eoIdx(j);

            if ~is_pair_allowed(AssocMatrix, Nr, Ne, Nm, ...
                    'Radar', rid, 'EO', eid)
                continue;
            end

            eMeas = squeeze(AlignedObs.EO.Y(k, eid, :)).';

            eAz = eMeas(1);
            eEl = eMeas(2);

            dAz = wrap_pi(eAz - rAz);
            dEl = wrap_pi(eEl - rEl);

            angleCost = sqrt(dAz^2 + dEl^2);

            if angleCost > gateEO
                continue;
            end

            rTopo = topology_vector( ...
                AlignedObs.Radar.Y, k, rid, radarIdx, [2 3], topoK);

            eTopo = topology_vector( ...
                AlignedObs.EO.Y, k, eid, eoIdx, [1 2], topoK);

            topoCost = norm(rTopo - eTopo) / sqrt(topoK);

            if topoCost > gateTopoEO
                continue;
            end

            cost = angleCost + lambdaTopoEO * topoCost;

            if ~isnan(prevEOForRadar(rid))
                if prevEOForRadar(rid) == eid
                    cost = 0.70 * cost;
                else
                    cost = cost + switchPenaltyEO;
                end
            end

            if cost <= totalGateEO
                C_RE(i, j) = cost;
            end
        end
    end

    C_RE = reject_ambiguous_rows(C_RE, ambDiffEO);
    pairsRE = greedy_assignment(C_RE, totalGateEO);

    %% --------------------------------------------------------------------
    % 4.2 Radar-ESM 关联代价矩阵
    %
    % 关键：
    %   Radar [range, az, el] -> 世界坐标
    %   世界坐标 -> ESM 所在平台机体系
    %   再比较预测 ESM 方位角与 ESM 实测方位角
    %% --------------------------------------------------------------------
    C_RM = inf(length(radarIdx), length(esmIdx));

    for i = 1:length(radarIdx)

        rid = radarIdx(i);
        rMeas = squeeze(AlignedObs.Radar.Y(k, rid, :)).';

        rRange = rMeas(1);
        rAz    = rMeas(2);
        rEl    = rMeas(3);

        radarUavID = get_track_uav_id(AlignedObs.Radar, rid, 1);

        pRadarUAV = local_get_uav_pos(AlignedObs, k, radarUavID);
        eRadarUAV = local_get_uav_euler(AlignedObs, k, radarUavID);

        pTargetWorld = local_radar_meas_to_world( ...
            rRange, rAz, rEl, pRadarUAV, eRadarUAV);

        for j = 1:length(esmIdx)

            mid = esmIdx(j);

            if ~is_pair_allowed(AssocMatrix, Nr, Ne, Nm, ...
                    'Radar', rid, 'ESM', mid)
                continue;
            end

            esmUavID = get_track_uav_id(AlignedObs.ESM, mid, 1);

            pESMUAV = local_get_uav_pos(AlignedObs, k, esmUavID);
            eESMUAV = local_get_uav_euler(AlignedObs, k, esmUavID);

            mAzPred = local_world_to_platform_az( ...
                pTargetWorld, pESMUAV, eESMUAV);

            mAz = squeeze(AlignedObs.ESM.Y(k, mid, :));

            angleCost = abs(wrap_pi(mAz - mAzPred));

            if angleCost > gateESM
                continue;
            end

            rTopo = topology_vector( ...
                AlignedObs.Radar.Y, k, rid, radarIdx, [2], topoK);

            mTopo = topology_vector( ...
                AlignedObs.ESM.Y, k, mid, esmIdx, [1], topoK);

            topoCost = norm(rTopo - mTopo) / sqrt(topoK);

            if topoCost > gateTopoESM
                continue;
            end

            cost = angleCost + lambdaTopoESM * topoCost;

            if ~isnan(prevESMForRadar(rid))
                if prevESMForRadar(rid) == mid
                    cost = 0.70 * cost;
                else
                    cost = cost + switchPenaltyESM;
                end
            end

            if cost <= totalGateESM
                C_RM(i, j) = cost;
            end
        end
    end

    C_RM = reject_ambiguous_rows(C_RM, ambDiffESM);
    pairsRM = greedy_assignment(C_RM, totalGateESM);

    %% --------------------------------------------------------------------
    % 4.3 生成关联结果
    %% --------------------------------------------------------------------
    for i = 1:length(radarIdx)

        rid = radarIdx(i);

        radarMeas = squeeze(AlignedObs.Radar.Y(k, rid, :)).';

        eoID = NaN;
        esmID = NaN;

        eoMeas = [];
        esmMeas = [];

        costEO = NaN;
        costESM = NaN;

        idxPairEO = find(pairsRE(:, 1) == i, 1);

        if ~isempty(idxPairEO)

            eoLocalCol = pairsRE(idxPairEO, 2);
            candidateEO = eoIdx(eoLocalCol);

            if is_pair_allowed(AssocMatrix, Nr, Ne, Nm, ...
                    'Radar', rid, 'EO', candidateEO)

                eoID = candidateEO;
                eoMeas = squeeze(AlignedObs.EO.Y(k, eoID, :)).';
                costEO = pairsRE(idxPairEO, 3);

                prevEOForRadar(rid) = eoID;
            end
        end

        idxPairESM = find(pairsRM(:, 1) == i, 1);

        if ~isempty(idxPairESM)

            esmLocalCol = pairsRM(idxPairESM, 2);
            candidateESM = esmIdx(esmLocalCol);

            if is_pair_allowed(AssocMatrix, Nr, Ne, Nm, ...
                    'Radar', rid, 'ESM', candidateESM)

                esmID = candidateESM;
                esmMeas = squeeze(AlignedObs.ESM.Y(k, esmID, :)).';
                costESM = pairsRM(idxPairESM, 3);

                prevESMForRadar(rid) = esmID;
            end
        end

        if ~isnan(eoID) && ~isnan(esmID)

            allowed_EO_ESM = is_pair_allowed(AssocMatrix, Nr, Ne, Nm, ...
                'EO', eoID, 'ESM', esmID);

            if ~allowed_EO_ESM
                if costEO <= costESM
                    esmID = NaN;
                    esmMeas = [];
                    costESM = NaN;
                else
                    eoID = NaN;
                    eoMeas = [];
                    costEO = NaN;
                end
            end
        end

        tr.globalID = rid;
        tr.radarID = rid;
        tr.eoID = eoID;
        tr.esmID = esmID;

        tr.radarMeas = radarMeas;
        tr.eoMeas = eoMeas;
        tr.esmMeas = esmMeas;

        tr.costEO = costEO;
        tr.costESM = costESM;

        tr.radarTruthID = get_truth_id(AlignedObs.Radar.truthID, k, rid);
        tr.eoTruthID    = get_truth_id(AlignedObs.EO.truthID, k, eoID);
        tr.esmTruthID   = get_truth_id(AlignedObs.ESM.truthID, k, esmID);

        tr.sourceSet = build_source_set(rid, eoID, esmID);

        tr.globalRadarID = local_to_global_id(Nr, Ne, Nm, 'Radar', rid);
        tr.globalEOID    = local_to_global_id(Nr, Ne, Nm, 'EO', eoID);
        tr.globalESMID   = local_to_global_id(Nr, Ne, Nm, 'ESM', esmID);

        tracks(end+1) = tr; %#ok<AGROW>
    end

    AssocOut.Frame(k).time = tk;
    AssocOut.Frame(k).tracks = tracks;
    AssocOut.Frame(k).numTracks = length(tracks);
end

end

%% ========================================================================
function AssocParam = build_default_assoc_param(Nr, Ne, Nm)

Ntotal = Nr + Ne + Nm;

idxRadar = 1:Nr;
idxEO    = Nr + (1:Ne);
idxESM   = Nr + Ne + (1:Nm);

AssocMatrix = true(Ntotal, Ntotal);

AssocMatrix(1:Ntotal+1:end) = false;

AssocMatrix(idxRadar, idxRadar) = false;
AssocMatrix(idxEO, idxEO)       = false;
AssocMatrix(idxESM, idxESM)     = false;

AssocMatrix = AssocMatrix & AssocMatrix.';

AssocParam.AssocMatrix = AssocMatrix;
AssocParam.idxRadar = idxRadar;
AssocParam.idxEO = idxEO;
AssocParam.idxESM = idxESM;
AssocParam.Nr = Nr;
AssocParam.Ne = Ne;
AssocParam.Nm = Nm;
AssocParam.Ntotal = Ntotal;

end

%% ========================================================================
function allowed = is_pair_allowed(AssocMatrix, Nr, Ne, Nm, sourceA, idA, sourceB, idB)

if isnan(idA) || isnan(idB)
    allowed = false;
    return;
end

gidA = local_to_global_id(Nr, Ne, Nm, sourceA, idA);
gidB = local_to_global_id(Nr, Ne, Nm, sourceB, idB);

if isnan(gidA) || isnan(gidB)
    allowed = false;
    return;
end

allowed = logical(AssocMatrix(gidA, gidB));

end

%% ========================================================================
function gid = local_to_global_id(Nr, Ne, Nm, sourceName, localID)

if isnan(localID)
    gid = NaN;
    return;
end

switch sourceName

    case 'Radar'
        if localID < 1 || localID > Nr
            gid = NaN;
        else
            gid = localID;
        end

    case 'EO'
        if localID < 1 || localID > Ne
            gid = NaN;
        else
            gid = Nr + localID;
        end

    case 'ESM'
        if localID < 1 || localID > Nm
            gid = NaN;
        else
            gid = Nr + Ne + localID;
        end

    otherwise
        gid = NaN;
end

end

%% ========================================================================
function tracks = init_empty_tracks()

tracks = struct( ...
    'globalID', {}, ...
    'radarID', {}, ...
    'eoID', {}, ...
    'esmID', {}, ...
    'radarMeas', {}, ...
    'eoMeas', {}, ...
    'esmMeas', {}, ...
    'costEO', {}, ...
    'costESM', {}, ...
    'radarTruthID', {}, ...
    'eoTruthID', {}, ...
    'esmTruthID', {}, ...
    'sourceSet', {}, ...
    'globalRadarID', {}, ...
    'globalEOID', {}, ...
    'globalESMID', {});

end

%% ========================================================================
function id = get_truth_id(truthMat, k, localID)

if isnan(localID)
    id = NaN;
    return;
end

id = truthMat(k, localID);

if id == 0
    id = NaN;
end

end

%% ========================================================================
function sourceSet = build_source_set(radarID, eoID, esmID)

sourceSet = {};

if ~isnan(radarID)
    sourceSet{end+1} = sprintf('Radar-%d', radarID); %#ok<AGROW>
end

if ~isnan(eoID)
    sourceSet{end+1} = sprintf('EO-%d', eoID); %#ok<AGROW>
end

if ~isnan(esmID)
    sourceSet{end+1} = sprintf('ESM-%d', esmID); %#ok<AGROW>
end

end

%% ========================================================================
function v = topology_vector(Y, k, id, validIdx, dimIdx, topoK)

center = squeeze(Y(k, id, dimIdx)).';
center = center(:).';

dList = [];

for q = 1:length(validIdx)

    otherID = validIdx(q);

    if otherID == id
        continue;
    end

    other = squeeze(Y(k, otherID, dimIdx)).';
    other = other(:).';

    d = 0;

    for m = 1:length(dimIdx)
        d = d + wrap_pi(other(m) - center(m))^2;
    end

    dList(end+1, 1) = sqrt(d); %#ok<AGROW>
end

dList = sort(dList);

padValue = 0.12;

v = padValue * ones(1, topoK);

numUse = min(topoK, length(dList));

if numUse > 0
    v(1:numUse) = dList(1:numUse);
end

end

%% ========================================================================
function C2 = reject_ambiguous_rows(C, diffThreshold)

C2 = C;

for i = 1:size(C, 1)

    vals = C(i, :);
    vals = vals(isfinite(vals));
    vals = sort(vals);

    if length(vals) >= 2
        if vals(2) - vals(1) < diffThreshold
            C2(i, :) = inf;
        end
    end
end

end

%% ========================================================================
function pairs = greedy_assignment(C, gate)

pairs = zeros(0, 3);

if isempty(C)
    return;
end

Cwork = C;

while true

    [val, idx] = min(Cwork(:));

    if isempty(val) || ~isfinite(val) || val > gate
        break;
    end

    [r, c] = ind2sub(size(Cwork), idx);

    pairs(end+1, :) = [r, c, val]; %#ok<AGROW>

    Cwork(r, :) = inf;
    Cwork(:, c) = inf;
end

end

%% ========================================================================
function y = wrap_pi(x)

y = mod(x + pi, 2*pi) - pi;

end

%% ========================================================================
function uavID = get_track_uav_id(SensorStruct, localID, defaultID)

uavID = defaultID;

if isfield(SensorStruct, 'uavID') && localID <= length(SensorStruct.uavID)
    tempID = SensorStruct.uavID(localID);

    if isfinite(tempID) && tempID >= 1
        uavID = tempID;
    end
end

end

%% ========================================================================
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

%% ========================================================================
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
function az = local_world_to_platform_az(pWorld, pUAV, euler)

Rwb = local_rpy2rot(euler);

pRelWorld = pWorld(:) - pUAV(:);
pRelBody = Rwb' * pRelWorld;

x = pRelBody(1);
y = pRelBody(2);

az = atan2(y, x);

end

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