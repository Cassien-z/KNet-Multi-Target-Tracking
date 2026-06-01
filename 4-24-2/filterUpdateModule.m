function [tracks, confusionCount, validCount] = filterUpdateModule( ...
    tracks, pred, frameData, assignment, costMat, ...
    sensor, confusionCount, validCount, k)
% =========================================================
% 滤波更新模块（增强版 + 朝圆心量测模型）
% 改进：
% 1. Radar 为主更新
% 2. EO / ESM 默认弱更新
% 3. 没有 Radar 时，不做强位置拉偏
% 4. 量测模型与数据生成、识别模块一致
% =========================================================

groups = frameData.groups;
nG = numel(groups);

radarOwn = frameData.radarPlatformState;
eoOwn    = frameData.eoPlatformState;
esmOwn   = frameData.esmPlatformState;
circleCenter = frameData.circleCenter;

numTargets = numel(tracks);

for j = 1:numTargets
    g = assignment(j);

    if g > 0 && g <= nG && costMat(j,g) < 1e5
        if isfield(groups(g), 'truthId')
            trueId = groups(g).truthId;
            if trueId >= 1 && trueId <= numTargets
                confusionCount(trueId, j, k) = confusionCount(trueId, j, k) + 1;
                validCount(trueId, k) = validCount(trueId, k) + 1;
            end
        end

        [xUpd, PUpd] = robustFilterUpdate( ...
            pred(j).x, pred(j).P, ...
            radarOwn, eoOwn, esmOwn, circleCenter, ...
            groups(g), sensor);

        tracks(j).x = xUpd;
        tracks(j).P = makeSPD(PUpd);
    else
        tracks(j).x = pred(j).x;
        tracks(j).P = makeSPD(pred(j).P);
    end

    tracks(j).kLast = k;
    tracks(j).history(:,k) = tracks(j).x;
end

end

%% =========================================================
% 本地函数
%% =========================================================
function [xUpd, PUpd] = robustFilterUpdate(xPred, PPred, radarOwn, eoOwn, esmOwn, circleCenter, group, sensor)
xUpd = xPred;
PUpd = PPred;

% ===== Radar 主更新 =====
if group.hasRadar
    [xUpd, PUpd] = iekfUpdate(xUpd, PUpd, group.zRadar, sensor.radar.R, ...
        @(x) hRadar(x, radarOwn, circleCenter), 3);

    % EO 弱更新
    if group.hasEO
        RweakEO = 6.0 * sensor.eo.R;
        [xUpd, PUpd] = iekfUpdate(xUpd, PUpd, group.zEO, RweakEO, ...
            @(x) hEO(x, eoOwn, circleCenter), 1);
    end

    % ESM 很弱更新
    if group.hasESM
        RweakESM = 8.0 * sensor.esm.R;
        [xUpd, PUpd] = iekfUpdate(xUpd, PUpd, group.zESM, RweakESM, ...
            @(x) hESM(x, esmOwn, circleCenter), 1);
    end

else
    % ===== 没有 Radar 时，保守更新 =====
    if group.hasEO
        RweakEO = 10.0 * sensor.eo.R;
        [xUpd, PUpd] = iekfUpdate(xUpd, PUpd, group.zEO, RweakEO, ...
            @(x) hEO(x, eoOwn, circleCenter), 1);
    end

    if group.hasESM
        RweakESM = 15.0 * sensor.esm.R;
        [xUpd, PUpd] = iekfUpdate(xUpd, PUpd, group.zESM, RweakESM, ...
            @(x) hESM(x, esmOwn, circleCenter), 1);
    end
end

end

function [xNew, PNew] = iekfUpdate(x, P, z, R, hfun, nIter)
xIter = x;

for it = 1:nIter
    H = numericalJacobian(hfun, xIter);
    h = hfun(xIter);
    v = z - h;

    if numel(v) >= 1
        v(1) = wrapToPiLocal(v(1));
    end
    if numel(v) >= 2 && (numel(v)==2 || numel(v)==3)
        v(2) = wrapToPiLocal(v(2));
    end

    S = makeSPD(H*P*H' + R);
    K = P*H'/S;
    xIter = x + K*(v - H*(x - xIter));
end

H = numericalJacobian(hfun, xIter);
h = hfun(xIter);
v = z - h;

if numel(v) >= 1
    v(1) = wrapToPiLocal(v(1));
end
if numel(v) >= 2 && (numel(v)==2 || numel(v)==3)
    v(2) = wrapToPiLocal(v(2));
end

S = makeSPD(H*P*H' + R);
K = P*H'/S;

% innovation gating，避免大残差把状态拉偏
nis = v' / S * v;
if nis > 25
    xNew = x;
    PNew = P;
    return;
end

xNew = x + K*v;
PNew = makeSPD(P - K*S*K');
end

function z = hRadar(x, own, circleCenter)
m = sensorMeasurementsPlatform(x, own, circleCenter);
z = m.radar;
end

function z = hEO(x, own, circleCenter)
m = sensorMeasurementsPlatform(x, own, circleCenter);
z = m.eo;
end

function z = hESM(x, own, circleCenter)
m = sensorMeasurementsPlatform(x, own, circleCenter);
z = m.esm;
end

function z = sensorMeasurementsPlatform(x6, own, circleCenter)
posT = x6(1:3);
posO = own(1:3);
velO = own(4:6);

Rib = inertialToBodyRotationLookCenter(posO, velO, circleCenter);
relI = posT - posO;
relB = Rib * relI;

xb = relB(1); yb = relB(2); zb = relB(3);
rr = sqrt(xb^2 + yb^2 + zb^2);
az = atan2(yb, xb);
el = atan2(zb, sqrt(xb^2 + yb^2));

z.radar = [az; el; rr];
z.eo    = [az; el];
z.esm   = az;
end

function Rib = inertialToBodyRotationLookCenter(posO, velO, circleCenter)
ex = circleCenter - posO;
if norm(ex) < 1e-6
    ex = [1;0;0];
else
    ex = ex / norm(ex);
end

v = velO(:);
if norm(v) < 1e-6
    v = [0;1;0];
else
    v = v / norm(v);
end

ez_tmp = [0;0;1];
ey = cross(ez_tmp, ex);
if norm(ey) < 1e-6
    ey = cross(v, ex);
end
if norm(ey) < 1e-6
    ey = [0;1;0];
else
    ey = ey / norm(ey);
end

ez = cross(ex, ey);
ez = ez / norm(ez);

Rbi = [ex(:), ey(:), ez(:)];
Rib = Rbi';
end

function H = numericalJacobian(fun, x)
fx = fun(x);
m = numel(fx);
n = numel(x);
H = zeros(m, n);
eps0 = 1e-5;

for i = 1:n
    dx = zeros(n,1);
    dx(i) = eps0;
    fp = fun(x + dx);
    fm = fun(x - dx);
    H(:,i) = (fp - fm) / (2*eps0);
end
end

function A = makeSPD(A)
A = 0.5*(A + A');
[V,D] = eig(A);
d = diag(D);
d(d < 1e-8) = 1e-8;
A = V * diag(d) * V';
A = 0.5*(A + A');
end

function ang = wrapToPiLocal(ang)
ang = mod(ang + pi, 2*pi) - pi;
end