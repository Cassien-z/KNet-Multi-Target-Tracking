function [assignment, costMat] = recognitionModule(pred, frameData, sensor, gate)
% =========================================================
% 识别/关联模块（更真实版）
% 目标：
% 1) 不再过于理想
% 2) Radar 仍主导
% 3) 历史轨迹与速度趋势仍参与
% 4) 但不过度“锁死”
% =========================================================

numTargets = numel(pred);
groups = frameData.groups;
nG = numel(groups);

costMat = 1e6 * ones(numTargets, nG);

radarOwn = frameData.radarPlatformState;
eoOwn    = frameData.eoPlatformState;
esmOwn   = frameData.esmPlatformState;
circleCenter = frameData.circleCenter;

for j = 1:numTargets
    for g = 1:nG
        [valid, cost] = singleGroupCost( ...
            pred(j), radarOwn, eoOwn, esmOwn, circleCenter, ...
            groups(g), sensor, gate);
        if valid
            costMat(j,g) = cost;
        end
    end
end

assignment = solveAssignmentBruteForce(costMat);

end

%% =========================================================
% 本地函数
%% =========================================================
function [valid, cost] = singleGroupCost(predj, radarOwn, eoOwn, esmOwn, circleCenter, group, sensor, gate)
valid = false;
cost = 1e6;

xPred = predj.x;
PPred = predj.P;
posHistPred = predj.posHistPred;

J = 0;
hasStrongObs = false;

% -------- Radar 主导 --------
if group.hasRadar
    h = hRadar(xPred, radarOwn, circleCenter);
    H = numericalJacobian(@(xx) hRadar(xx, radarOwn, circleCenter), xPred);
    v = group.zRadar - h;
    v(1) = wrapToPiLocal(v(1));
    v(2) = wrapToPiLocal(v(2));

    S = makeSPD(H*PPred*H' + sensor.radar.R);
    d2 = v' / S * v;
    if d2 > gate.radar
        return;
    end

    J = J + 1.0 * d2;
    hasStrongObs = true;

    posMeas = radarMeasurementToCartesian(group.zRadar, radarOwn, circleCenter);
    posErr = norm(posMeas - posHistPred);

    % 历史位置一致性项：比“100%版本”弱一点
    J = J + 0.0045 * posErr^2;

    velPred = xPred(4:6);
    dirPred = velPred / max(norm(velPred), 1e-6);
    dirMeas = posMeas - xPred(1:3);
    dirMeas = dirMeas / max(norm(dirMeas), 1e-6);
    trendPenalty = 1 - dot(dirPred, dirMeas);
    J = J + 0.6 * max(0, trendPenalty);
end

% -------- EO 辅助 --------
if group.hasEO
    h = hEO(xPred, eoOwn, circleCenter);
    H = numericalJacobian(@(xx) hEO(xx, eoOwn, circleCenter), xPred);
    v = group.zEO - h;
    v(1) = wrapToPiLocal(v(1));
    v(2) = wrapToPiLocal(v(2));

    S = makeSPD(H*PPred*H' + sensor.eo.R);
    d2 = v' / S * v;
    if d2 > gate.eo
        return;
    end

    J = J + 0.30 * d2;
end

% -------- ESM 弱辅助 --------
if group.hasESM
    h = hESM(xPred, esmOwn, circleCenter);
    H = numericalJacobian(@(xx) hESM(xx, esmOwn, circleCenter), xPred);
    v = wrapToPiLocal(group.zESM - h);

    S = H*PPred*H' + sensor.esm.R;
    S = max(S, 1e-8);
    d2 = v' / S * v;
    if d2 > gate.esm
        return;
    end

    J = J + 0.10 * d2;
end

% 纯 ESM 不参与硬关联
if group.hasESM && ~group.hasRadar && ~group.hasEO
    return;
end

% 只有 EO 时更严格，但不至于太苛刻
if group.hasEO && ~group.hasRadar
    eoPred = hEO(xPred, eoOwn, circleCenter);
    dv = group.zEO - eoPred;
    dv(1) = wrapToPiLocal(dv(1));
    dv(2) = wrapToPiLocal(dv(2));
    if norm(dv) > 0.028
        return;
    end
end

if group.hasRadar || (group.hasEO && group.hasESM)
    hasStrongObs = true;
end

if ~hasStrongObs
    return;
end

if J < gate.total
    valid = true;
    cost = J;
end
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

function posI = radarMeasurementToCartesian(zRadar, own, circleCenter)
az = zRadar(1); el = zRadar(2); rr = zRadar(3);

xb = rr*cos(el)*cos(az);
yb = rr*cos(el)*sin(az);
zb = rr*sin(el);

relB = [xb; yb; zb];
Rib = inertialToBodyRotationLookCenter(own(1:3), own(4:6), circleCenter);
Rbi = Rib';
posI = own(1:3) + Rbi * relB;
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

function assignment = solveAssignmentBruteForce(costMat)
[nTrack, nMeas] = size(costMat);
assignment = zeros(1, nTrack);

if nMeas == 0
    return;
end

bestCost = inf;
bestAssign = zeros(1, nTrack);
currentAssign = zeros(1, nTrack);
usedCols = false(1, nMeas);

search(1, 0);
assignment = bestAssign;

    function search(iTrack, currentCost)
        if iTrack > nTrack
            if currentCost < bestCost
                bestCost = currentCost;
                bestAssign = currentAssign;
            end
            return;
        end

        currentAssign(iTrack) = 0;
        search(iTrack + 1, currentCost + 20);

        for j = 1:nMeas
            if ~usedCols(j)
                c = costMat(iTrack, j);
                if c < 1e5
                    usedCols(j) = true;
                    currentAssign(iTrack) = j;
                    search(iTrack + 1, currentCost + c);
                    usedCols(j) = false;
                end
            end
        end
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