function [measData, truth] = buildExampleMeasData()
% =========================================================
% 生成示例输入数据 measData（更真实版）
% 关键改进：
% 1) 三架无人机在同一个圆上运动
% 2) 初始位置均匀分布
% 3) 每架无人机各转 60°
% 4) 传感器朝圆心观测
% =========================================================

rng('shuffle');

T = 90;
dt = 1.0;
numTargets = 3;

sensor.radar.sigma_ang = 0.002;
sensor.radar.sigma_r   = 80;
sensor.radar.azFOV = deg2rad(60);
sensor.radar.elFOV = deg2rad(3);
sensor.radar.Pd    = 0.95;

sensor.eo.sigma_ang = 0.002;
sensor.eo.azFOV = deg2rad(60);
sensor.eo.elFOV = deg2rad(3);
sensor.eo.Pd    = 0.92;

sensor.esm.sigma_az = 0.025;
sensor.esm.Pd = 0.98;

scene.rangeMin = 500;
scene.rangeMax = 12000;
lambdaClutterGroups = 4;

[targetTruth, ownTruth, circleCenter] = generateScenario(T, dt, numTargets);

measData = struct([]);

for k = 1:T
    radarOwn = ownTruth(:,k,1);
    eoOwn    = ownTruth(:,k,2);
    esmOwn   = ownTruth(:,k,3);

    groups = generateMeasurementGroups( ...
        targetTruth(:,k,:), radarOwn, eoOwn, esmOwn, ...
        sensor, scene, lambdaClutterGroups, circleCenter);

    measData(k).radarPlatformState = radarOwn;
    measData(k).eoPlatformState    = eoOwn;
    measData(k).esmPlatformState   = esmOwn;
    measData(k).groups             = groups;
    measData(k).circleCenter       = circleCenter;
end

truth = struct;
truth.targetTruth = targetTruth;
truth.ownTruth = ownTruth;
truth.circleCenter = circleCenter;
truth.numTargets = numTargets;

end

%% =========================================================
% 场景生成
%% =========================================================
function [targetTruth, ownTruth, center] = generateScenario(T, dt, numTargets)
targetTruth = zeros(6, T, numTargets);
ownTruth    = zeros(6, T, 3);

center = [0; -1800; 1050];

%% ================= 三架平台：同圆、均匀分布、各转60° =================
R0 = 2200;

% 三架无人机在圆上的初始角位置均匀分布
startAngles = deg2rad([0, 120, 240]);

% 每架无人机都只飞 60°
arcSpan = deg2rad(60);

% 匀速转过这 60°
omega = arcSpan / ((T-1) * dt);

for p = 1:3
    theta0 = startAngles(p);

    for k = 1:T
        theta = theta0 + (k-1) * omega;

        px = center(1) + R0*cos(theta);
        py = center(2) + R0*sin(theta);
        pz = center(3);

        vx = -R0*omega*sin(theta);
        vy =  R0*omega*cos(theta);
        vz = 0;

        ownTruth(:,k,p) = [px; py; pz; vx; vy; vz];
    end
end

%% ================= 三个目标轨迹 =================
targetTruth(:,1,1) = [center + [-760;  230;  -6];
                      28;  5;  0.01];

targetTruth(:,1,2) = [center + [ 780; -180;   4];
                     -25;  7; -0.01];

targetTruth(:,1,3) = [center + [ 110;  900;   5];
                       3; -27;  0.01];

F6 = [1 0 0 dt 0  0;
      0 1 0 0  dt 0;
      0 0 1 0  0  dt;
      0 0 0 1  0  0;
      0 0 0 0  1  0;
      0 0 0 0  0  1];

G = [0.5*dt^2 0        0;
     0        0.5*dt^2 0;
     0        0        0.5*dt^2;
     dt       0        0;
     0        dt       0;
     0        0        dt];

for k = 2:T
    t = k;
    for i = 1:numTargets
        switch i
            case 1
                a = [0.045*sin(0.07*t);
                     0.025*cos(0.05*t);
                     0.0010*sin(0.06*t)];
                if k > 18 && k < 30
                    a = a + [0.08; -0.05; 0];
                elseif k > 30 && k < 40
                    a = a + [-0.05; 0.03; 0];
                end

            case 2
                a = [-0.045*cos(0.06*t);
                      0.025*sin(0.05*t);
                      0.0010*cos(0.07*t)];
                if k > 36 && k < 50
                    a = a + [-0.08; 0.05; 0];
                elseif k > 50 && k < 60
                    a = a + [0.05; -0.03; 0];
                end

            case 3
                a = [0.018*cos(0.05*t);
                    -0.045*sin(0.06*t);
                     0.0010*cos(0.05*t)];
                if k > 58 && k < 74
                    a = a + [0.04; -0.08; 0];
                elseif k > 74 && k < 84
                    a = a + [-0.03; 0.05; 0];
                end
        end

        targetTruth(:,k,i) = F6 * targetTruth(:,k-1,i) + G * a;
    end
end
end

%% =========================================================
% 候选观测组
%% =========================================================
function groups = generateMeasurementGroups(targetSlice, radarOwn, eoOwn, esmOwn, sensor, scene, lambdaClutterGroups, circleCenter)
numTargets = size(targetSlice, 3);
groupCell = {};

for i = 1:numTargets
    x = targetSlice(:,1,i);
    g = createEmptyGroup();
    g.truthId = i;

    zRadarTrue = sensorMeasurementsPlatform(x, radarOwn, circleCenter);
    zEOTrue    = sensorMeasurementsPlatform(x, eoOwn, circleCenter);
    zESMTrue   = sensorMeasurementsPlatform(x, esmOwn, circleCenter);

    if inRadarFOV(zRadarTrue, sensor) && rand <= sensor.radar.Pd
        g.hasRadar = true;
        g.zRadar = zRadarTrue.radar + [sensor.radar.sigma_ang * randn;
                                       sensor.radar.sigma_ang * randn;
                                       sensor.radar.sigma_r   * randn];
    end

    if inEOFOV(zEOTrue, sensor) && rand <= sensor.eo.Pd
        g.hasEO = true;
        g.zEO = zEOTrue.eo + [sensor.eo.sigma_ang * randn;
                              sensor.eo.sigma_ang * randn];
    end

    if rand <= sensor.esm.Pd
        g.hasESM = true;
        g.zESM = zESMTrue.esm + sensor.esm.sigma_az * randn;
    end

    if g.hasRadar || g.hasEO || g.hasESM
        groupCell{end+1} = g; %#ok<AGROW>
    end
end

M = poissrnd_local(lambdaClutterGroups);
for m = 1:M
    g = createEmptyGroup();
    g.truthId = 0;

    useRadar = rand < 0.22;
    useEO    = rand < 0.15;
    useESM   = rand < 0.30;

    if ~(useRadar || useEO || useESM)
        useESM = true;
    end

    if useRadar
        az = (2*rand - 1) * sensor.radar.azFOV;
        el = (2*rand - 1) * sensor.radar.elFOV;
        rr = scene.rangeMin + (scene.rangeMax - scene.rangeMin) * rand;
        g.hasRadar = true;
        g.zRadar = [az; el; rr];
    end

    if useEO
        az = (2*rand - 1) * sensor.eo.azFOV;
        el = (2*rand - 1) * sensor.eo.elFOV;
        g.hasEO = true;
        g.zEO = [az; el];
    end

    if useESM
        az = -pi + 2*pi*rand;
        g.hasESM = true;
        g.zESM = az;
    end

    groupCell{end+1} = g; %#ok<AGROW>
end

if isempty(groupCell)
    groups = repmat(createEmptyGroup(), 0, 1);
else
    groups = [groupCell{:}];
end
end

function g = createEmptyGroup()
g = struct('hasRadar', false, 'hasEO', false, 'hasESM', false, ...
           'zRadar', [], 'zEO', [], 'zESM', [], 'truthId', 0);
end

%% =========================================================
% 量测模型：朝圆心
%% =========================================================
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

function tf = inRadarFOV(zTrue, sensor)
tf = (abs(zTrue.radar(1)) <= sensor.radar.azFOV) && ...
     (abs(zTrue.radar(2)) <= sensor.radar.elFOV);
end

function tf = inEOFOV(zTrue, sensor)
tf = (abs(zTrue.eo(1)) <= sensor.eo.azFOV) && ...
     (abs(zTrue.eo(2)) <= sensor.eo.elFOV);
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

function n = poissrnd_local(lambda)
L = exp(-lambda);
p = 1;
n = 0;
while p > L
    n = n + 1;
    p = p * rand;
end
n = n - 1;
end