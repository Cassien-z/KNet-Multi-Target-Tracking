function pred = predictTracks(tracks, model)
% =========================================================
% 预测模块
% 输入：
%   tracks : 上一时刻航迹
%   model  : 动态模型参数
% 输出：
%   pred(j).x
%   pred(j).P
%   pred(j).posHistPred
% =========================================================

numTargets = numel(tracks);
pred = struct([]);

for i = 1:numTargets
    xCV = model.F * tracks(i).x;
    PCV = model.F * tracks(i).P * model.F' + model.Q;

    k = tracks(i).kLast;
    posHistPred = xCV(1:3);

    % 历史轨迹外推
    if k >= 3 && all(~isnan(tracks(i).history(1,k-2:k)))
        p1 = tracks(i).history(1:3,k);
        p2 = tracks(i).history(1:3,k-1);
        p3 = tracks(i).history(1:3,k-2);

        v1 = p1 - p2;
        v0 = p2 - p3;
        aEst = v1 - v0;
        pHist = p1 + v1 + 0.5*aEst;

        alpha = 0.35;
        posHistPred = pHist;
        xCV(1:3) = (1-alpha)*xCV(1:3) + alpha*pHist;
    end

    pred(i).x = xCV;
    pred(i).P = makeSPD(PCV);
    pred(i).posHistPred = posHistPred;
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