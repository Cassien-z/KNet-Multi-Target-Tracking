function prediction = run_tracker(modelPath, input)
%RUN_TRACKER Execute the fixed-shape end-to-end tracker from MATLAB.
%   PREDICTION = RUN_TRACKER(MODELPATH, INPUT) calls the packaged MEX/ONNX
%   Runtime implementation and returns a single [1,100,6] array. INPUT must
%   contain the 18 fields documented in portable/INTERFACE.md.

inputNames = [ ...
    "radar", "eo", "esm", "radar_mask", "eo_mask", "esm_mask", ...
    "ref_pos", "rotmat_b2e", "uav_pos", "fast_link", "slow_link", ...
    "fast_update", "slow_update", "imu", "fast_age", "slow_age", ...
    "esm_mode", "dt"];
inputCounts = [ ...
    1500, 1000, 1000, 500, 500, 500, 3, 4500, 1500, ...
    1500, 1500, 500, 500, 3000, 500, 500, 500, 1];

matlabDir = fileparts(mfilename("fullpath"));
if exist(fullfile(matlabDir, "coop_tracker_mex." + mexext), "file") == 0
    error("Tracker MEX is missing. Run build_mex from portable/matlab first.");
end

arguments = cell(1, numel(inputNames));
for index = 1:numel(inputNames)
    name = inputNames(index);
    fieldName = char(name);
    if ~isfield(input, fieldName)
        error("Missing tracker input field: %s", name);
    end
    value = single(input.(fieldName));
    if numel(value) ~= inputCounts(index)
        error("Input %s has %d elements; expected %d.", ...
            name, numel(value), inputCounts(index));
    end
    % ONNX Runtime expects C row-major buffers. Reverse MATLAB dimensions
    % before linearizing to preserve the public logical tensor layout.
    order = ndims(value):-1:1;
    value = permute(value, order);
    arguments{index} = value(:);
end

flatPrediction = coop_tracker_mex(char(string(modelPath)), arguments{:});
prediction = permute(reshape(flatPrediction, [6, 100, 1]), [3, 2, 1]);
prediction = single(prediction);
end
