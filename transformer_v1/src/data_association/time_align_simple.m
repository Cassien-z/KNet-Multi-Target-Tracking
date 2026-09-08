function [Y_align, validFlag] = time_align_simple(Data_Input, X_pred, t_pred, t_grid, uavID, targetID, sensorType)
%TIME_ALIGN_SIMPLE 简化版时间尺度对齐函数
%
% 输入：
%   Data_Input : 原始数据结构
%   X_pred     : P×3 预测状态，[X,Y,Z]，世界坐标
%   t_pred     : P×1 预测状态时间戳
%   t_grid     : K×1 统一对齐时标
%   uavID      : 无人机编号，MATLAB 下标
%   targetID   : 目标编号，MATLAB 下标
%   sensorType : 传感器类型，1=Radar，2=EO，3=ESM
%
% 输出：
%   Y_align   : K×D 对齐后的同步量测
%   validFlag : K×1，有效标志

%% ============================================================
% 固定参数
% ============================================================
win = 0.5;
minPts = 2;
predWeight = 0.6;
realWeight = 1.0;

t_pred = t_pred(:);
t_grid = t_grid(:);

%% ============================================================
% 传感器设置
% ============================================================
switch sensorType
    case 1
        sensorName = 'Radar';
        D = 3;
        angleIdx = [2 3];  % [range, az, el]
    case 2
        sensorName = 'EO';
        D = 2;
        angleIdx = [1 2];  % [az, el]
    case 3
        sensorName = 'ESM';
        D = 1;
        angleIdx = 1;      % [az]
    otherwise
        error('sensorType 必须为 1=Radar, 2=EO, 3=ESM');
end

%% ============================================================
% 1. 提取历史真实量测
% ============================================================
[t_meas, Y_meas] = get_history_meas(Data_Input, uavID, targetID, sensorName, sensorType);

%% ============================================================
% 2. 提取 UAV 平台轨迹
% ============================================================
[ts_uav, pos_uav, euler_uav] = extract_uav_pose(Data_Input, uavID);

%% ============================================================
% 3. 预测状态代入传感器量测方程
% ============================================================
Y_pred = zeros(length(t_pred), D);

for i = 1:length(t_pred)
    tp = t_pred(i);
    pTarget = X_pred(i, 1:3);

    pUAV = interp1(ts_uav, pos_uav, tp, 'linear', 'extrap');
    euler = interp1(ts_uav, euler_uav, tp, 'linear', 'extrap');

    Rwb = rpy2rot(euler);

    pRelWorld = pTarget(:) - pUAV(:);
    pRelBody = Rwb' * pRelWorld;

    Y_pred(i, :) = sensor_h(pRelBody, sensorType);
end

%% ============================================================
% 4. 合并真实量测和预测量测
% ============================================================
t_all = [t_meas; t_pred];
Y_all = [Y_meas; Y_pred];

w_all = [realWeight * ones(length(t_meas), 1); ...
         predWeight * ones(length(t_pred), 1)];

if isempty(t_all)
    Y_align = nan(length(t_grid), D);
    validFlag = false(length(t_grid), 1);
    return;
end

[t_all, idx] = sort(t_all);
Y_all = Y_all(idx, :);
w_all = w_all(idx);

% 角度展开
for j = angleIdx
    Y_all(:, j) = unwrap(Y_all(:, j));
end

%% ============================================================
% 5. 局部线性最小二乘时间对齐
% ============================================================
K = length(t_grid);
Y_align = nan(K, D);
validFlag = false(K, 1);

for k = 1:K
    tk = t_grid(k);

    idxWin = abs(t_all - tk) <= win;

    if sum(idxWin) < minPts
        continue;
    end

    tw = t_all(idxWin);
    Yw = Y_all(idxWin, :);
    ww = w_all(idxWin);

    tau = tw - tk;

    A = [ones(length(tau), 1), tau];

    W = diag(ww);

    beta = (A' * W * A) \ (A' * W * Yw);

    Y_align(k, :) = beta(1, :);
    validFlag(k) = true;
end

% 角度 wrap 回 [-pi, pi]
for j = angleIdx
    Y_align(:, j) = wrap_pi(Y_align(:, j));
end

end

%% ========================================================================
% 提取历史真实量测
% ========================================================================
function [t_meas, Y_meas] = get_history_meas(Data_Input, uavID, targetID, sensorName, sensorType)

t_meas = [];
Y_meas = [];

for k = 1:length(Data_Input)

    uav = Data_Input(k).UAV(uavID);

    if ~isfield(uav, sensorName)
        continue;
    end

    S = uav.(sensorName);

    if ~isfield(S, 'Meas') || ~isfield(S, 'Mask') || ~isfield(S, 'MeasTime')
        continue;
    end

    if targetID > length(S.Mask) || S.Mask(targetID) < 0.5
        continue;
    end

    raw = S.Meas(targetID, :);

    switch sensorType
        case 1
            % Radar.Meas = [azimuth, elevation, range]
            % 内部统一为 [range, azimuth, elevation]
            y = [raw(3), raw(1), raw(2)];

        case 2
            % EO.Meas = [azimuth, elevation]
            y = [raw(1), raw(2)];

        case 3
            % ESM.Meas = [azimuth, elevation]，这里只用 azimuth
            y = raw(1);
    end

    t_meas(end+1, 1) = S.MeasTime; %#ok<AGROW>
    Y_meas(end+1, :) = y;          %#ok<AGROW>
end

end

%% ========================================================================
% 提取 UAV 位姿轨迹
% ========================================================================
function [ts, pos, euler] = extract_uav_pose(Data_Input, uavID)

K = length(Data_Input);
ts = zeros(K, 1);
pos = zeros(K, 3);
euler = zeros(K, 3);

for k = 1:K
    ts(k) = Data_Input(k).GlobalTime;

    uav = Data_Input(k).UAV(uavID);

    pos(k, :) = uav.Pose.PositionWorld(:).';
    euler(k, :) = uav.Pose.EulerRPY(:).';
end

end

%% ========================================================================
% 传感器量测方程
% ========================================================================
function y = sensor_h(pRelBody, sensorType)

x  = pRelBody(1);
yy = pRelBody(2);
z  = pRelBody(3);

range = sqrt(x^2 + yy^2 + z^2);
az = atan2(yy, x);
el = atan2(z, sqrt(x^2 + yy^2));

switch sensorType
    case 1
        y = [range, az, el];
    case 2
        y = [az, el];
    case 3
        y = az;
end

end

%% ========================================================================
% RPY 转旋转矩阵，body -> world
% ========================================================================
function R = rpy2rot(euler)

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

%% ========================================================================
% wrap 到 [-pi, pi]
% ========================================================================
function y = wrap_pi(x)

y = mod(x + pi, 2*pi) - pi;

end