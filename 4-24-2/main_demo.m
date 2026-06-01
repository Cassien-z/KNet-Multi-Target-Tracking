clc; clear; close all;

%% =========================================================
% main_demo.m
%
% 功能：
% 1) 用 Monte Carlo 统计 probTensor(i,j,k)
% 2) 统计结束后，单独跑一次演示轮用于画图
% 3) 演示轮与统计轮分开，避免“图和统计不一致”
%
% 输出：
%   probTensor(i,j,k): 第k时刻将真实目标i识别成目标j的概率
%% =========================================================

rng(1);

MC = 300;
showPlot = true;

confusionCountTotal = [];
validCountTotal = [];

statTruth = [];
statMeasData = [];

%% =========================================================
% 第一部分：Monte Carlo 统计
%% =========================================================
for mc = 1:MC
    fprintf('Monte Carlo: %d / %d\n', mc, MC);

    [measData, truth] = buildExampleMeasData();

    numTargets = truth.numTargets;
    T = numel(measData);

    [sensor, gate, model] = getSystemConfig();

    % 初始化航迹（统计轮）
    tracks = initTracksFromTruth(truth, T, 'stat');

    confusionCount = zeros(numTargets, numTargets, T);
    validCount = zeros(numTargets, T);

    for k = 2:T
        pred = predictTracks(tracks, model);

        [assignment, costMat] = recognitionModule(pred, measData(k), sensor, gate);

        [tracks, confusionCount, validCount] = filterUpdateModule( ...
            tracks, pred, measData(k), assignment, costMat, ...
            sensor, confusionCount, validCount, k);
    end

    if mc == 1
        confusionCountTotal = zeros(size(confusionCount));
        validCountTotal = zeros(size(validCount));
        statTruth = truth;
        statMeasData = measData;
    end

    confusionCountTotal = confusionCountTotal + confusionCount;
    validCountTotal = validCountTotal + validCount;
end

%% 最终概率张量
probTensor = NaN(numTargets, numTargets, T);
for k = 1:T
    for i = 1:numTargets
        if validCountTotal(i,k) > 0
            probTensor(i,:,k) = confusionCountTotal(i,:,k) / validCountTotal(i,k);
        end
    end
end

disp(' ');
disp('第 20 个时刻的混淆概率矩阵 probTensor(:,:,20) = ');
disp(probTensor(:,:,20));

fprintf('\n================ Monte Carlo 平均结果 ================\n');
for i = 1:numTargets
    pii = squeeze(probTensor(i,i,:));
    validIdx = ~isnan(pii);
    if any(validIdx)
        fprintf('目标 %d 平均正确识别率: %.2f%%\n', i, mean(pii(validIdx))*100);
    else
        fprintf('目标 %d 平均正确识别率: NaN\n', i);
    end
end

fprintf('\n================ 有效识别覆盖率 ================\n');
for i = 1:numTargets
    coverage = mean(validCountTotal(i,:) > 0) * 100;
    fprintf('目标 %d 有效识别覆盖率: %.2f%%\n', i, coverage);
end

%% =========================================================
% 第二部分：单独跑一次演示轮，用于画图
% 目的：图像展示单独对应一轮完整结果，不和 Monte Carlo 混淆
%% =========================================================
fprintf('\n================ 开始单独演示轮 ================\n');

rng(2026);   % 固定一个单独的演示随机种子，保证图可复现
[demoMeasData, demoTruth] = buildExampleMeasData();
[sensor, gate, model] = getSystemConfig();

demoTracks = initTracksFromTruth(demoTruth, numel(demoMeasData), 'demo');

demoAssign = cell(1, numel(demoMeasData));

for k = 2:numel(demoMeasData)
    pred = predictTracks(demoTracks, model);

    [assignment, costMat] = recognitionModule(pred, demoMeasData(k), sensor, gate);
    demoAssign{k} = assignment;

    dummyConfusion = zeros(demoTruth.numTargets, demoTruth.numTargets, numel(demoMeasData));
    dummyValid = zeros(demoTruth.numTargets, numel(demoMeasData));

    [demoTracks, ~, ~] = filterUpdateModule( ...
        demoTracks, pred, demoMeasData(k), assignment, costMat, ...
        sensor, dummyConfusion, dummyValid, k);
end

%% 画图
if showPlot
    drawResults(demoMeasData, demoTruth, demoTracks, probTensor, validCountTotal, demoAssign);
end

%% =========================================================
% 本地函数
%% =========================================================

function [sensor, gate, model] = getSystemConfig()
    % 统一配置，统计轮和演示轮保持一致

    sensor.radar.sigma_ang = 0.002;
    sensor.radar.sigma_r   = 80;
    sensor.radar.R = diag([sensor.radar.sigma_ang^2,...
                           sensor.radar.sigma_ang^2,...
                           sensor.radar.sigma_r^2]);

    sensor.eo.sigma_ang = 0.002;
    sensor.eo.R = diag([sensor.eo.sigma_ang^2,...
                        sensor.eo.sigma_ang^2]);

    sensor.esm.sigma_az = 0.025;
    sensor.esm.R = sensor.esm.sigma_az^2;

    gate.radar = 16.0;
    gate.eo    = 16.0;
    gate.esm   = 16.0;
    gate.total = 17.5;

    model.dt = 1.0;
    dt = model.dt;

    model.F = [1 0 0 dt 0  0;
               0 1 0 0  dt 0;
               0 0 1 0  0  dt;
               0 0 0 1  0  0;
               0 0 0 0  1  0;
               0 0 0 0  0  1];

    q = 0.14;
    Q1 = [dt^4/4 0       0       dt^3/2 0       0;
          0      dt^4/4  0       0      dt^3/2  0;
          0      0       dt^4/4  0      0       dt^3/2;
          dt^3/2 0       0       dt^2   0       0;
          0      dt^3/2  0       0      dt^2    0;
          0      0       dt^3/2  0      0       dt^2];
    model.Q = q^2 * Q1;
end

function tracks = initTracksFromTruth(truth, T, mode)
    % 初始化航迹
    % mode='stat' 统计轮
    % mode='demo' 演示轮

    numTargets = truth.numTargets;
    tracks = struct([]);

    for i = 1:numTargets
        tracks(i).id = i;

        switch mode
            case 'stat'
                tracks(i).x = truth.targetTruth(:,1,i) + [20*randn;20*randn;8*randn;2.0*randn;2.0*randn;0.8*randn];
                tracks(i).P = diag([70^2,70^2,25^2,8^2,8^2,3^2]);

            case 'demo'
                % 演示轮可以比统计轮稍稳一点，方便看图，但不要太离谱
                tracks(i).x = truth.targetTruth(:,1,i) + [12*randn;12*randn;5*randn;1.2*randn;1.2*randn;0.5*randn];
                tracks(i).P = diag([45^2,45^2,18^2,5^2,5^2,2^2]);

            otherwise
                error('Unknown mode.');
        end

        tracks(i).history = nan(6, T);
        tracks(i).history(:,1) = tracks(i).x;
        tracks(i).kLast = 1;
    end
end

function drawResults(measData, truth, tracks, probTensor, validCountTotal, demoAssign)
    T = numel(measData);
    numTargets = numel(tracks);

    figure('Color','w','Position',[80 80 1000 720]);
    hold on; grid on; box on;

    platformColors = [0.85 0.10 0.10;
                      0.10 0.45 0.85;
                      0.10 0.60 0.20];
    targetColors = lines(numTargets);

    radarPath = zeros(6,T);
    eoPath    = zeros(6,T);
    esmPath   = zeros(6,T);

    for k = 1:T
        radarPath(:,k) = measData(k).radarPlatformState;
        eoPath(:,k)    = measData(k).eoPlatformState;
        esmPath(:,k)   = measData(k).esmPlatformState;
    end

    plot3(radarPath(1,:), radarPath(2,:), radarPath(3,:), ...
        '-', 'Color', platformColors(1,:), 'LineWidth', 2.2, ...
        'Marker', 'o', 'MarkerIndices', 1:10:T, 'MarkerSize', 5);

    plot3(eoPath(1,:), eoPath(2,:), eoPath(3,:), ...
        '-', 'Color', platformColors(2,:), 'LineWidth', 2.2, ...
        'Marker', '^', 'MarkerIndices', 1:10:T, 'MarkerSize', 5);

    plot3(esmPath(1,:), esmPath(2,:), esmPath(3,:), ...
        '-', 'Color', platformColors(3,:), 'LineWidth', 2.2, ...
        'Marker', 's', 'MarkerIndices', 1:10:T, 'MarkerSize', 5);

    plot3(truth.circleCenter(1), truth.circleCenter(2), truth.circleCenter(3), ...
        'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');

    for i = 1:numTargets
        plot3(squeeze(truth.targetTruth(1,:,i)), ...
              squeeze(truth.targetTruth(2,:,i)), ...
              squeeze(truth.targetTruth(3,:,i)), ...
              '-', 'LineWidth', 2.0, 'Color', targetColors(i,:));

        plot3(tracks(i).history(1,:), ...
              tracks(i).history(2,:), ...
              tracks(i).history(3,:), ...
              '--', 'LineWidth', 1.8, 'Color', targetColors(i,:));
    end

    xlabel('X / m');
    ylabel('Y / m');
    zlabel('Z / m');
    title('单独演示轮：三架无人机轨迹、真实轨迹与估计轨迹');
    legend({'飞机1(Radar)','飞机2(EO)','飞机3(ESM)','圆心', ...
            '目标1真实','目标1估计', ...
            '目标2真实','目标2估计', ...
            '目标3真实','目标3估计'}, ...
            'Location','bestoutside');
    view(135,25);

    figure('Color','w','Position',[120 120 1000 620]);
    hold on; grid on; box on;
    for i = 1:numTargets
        plot(1:T, squeeze(probTensor(i,i,:))*100, 'LineWidth', 2.4);
    end
    yline(80, 'm--', 'LineWidth', 2.0);
    xlabel('时间步');
    ylabel('P(i \rightarrow i) / %');
    title('各目标逐时刻正确识别概率（Monte Carlo 全局累计统计）');
    legend({'目标1->1','目标2->2','目标3->3','80%指标线'}, 'Location','best');
    ylim([0 105]);
    xlim([1 T]);

    figure('Color','w','Position',[220 180 560 460]);
    imagesc(probTensor(:,:,20)*100);
    colorbar;
    axis equal tight;
    xlabel('被识别成目标 j');
    ylabel('真实目标 i');
    title('第 20 个时刻的混淆概率矩阵 / %');
    set(gca, 'XTick', 1:numTargets, 'YTick', 1:numTargets);

    figure('Color','w','Position',[260 220 900 500]);
    hold on; grid on; box on;
    for i = 1:numTargets
        plot(1:T, (validCountTotal(i,:) > 0)*100, 'LineWidth', 1.8);
    end
    xlabel('时间步');
    ylabel('该时刻是否有有效识别 / %');
    title('有效识别覆盖情况');
    legend({'目标1','目标2','目标3'}, 'Location','best');
    ylim([-5 105]);
    xlim([1 T]);

    % 可选：演示轮某一时刻的 assignment 打印
    kShow = 20;
    if kShow <= numel(demoAssign) && ~isempty(demoAssign{kShow})
        fprintf('\n演示轮第 %d 时刻 assignment = \n', kShow);
        disp(demoAssign{kShow});
    end
end