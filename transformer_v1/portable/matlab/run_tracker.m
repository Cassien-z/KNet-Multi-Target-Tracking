function prediction = run_tracker(modelPath, input)
%RUN_TRACKER Execute the fixed-shape end-to-end tracker in MATLAB.
%   PREDICTION = RUN_TRACKER(MODELPATH, INPUT) loads the shared ONNX model
%   and returns a single-precision [1,100,6] state sequence. INPUT must
%   contain every field listed below with the shapes from manifest.json.

persistent net loadedModelPath

inputNames = [ ...
    "radar", "eo", "esm", "radar_mask", "eo_mask", "esm_mask", ...
    "ref_pos", "rotmat_b2e", "uav_pos", "fast_link", "slow_link", ...
    "fast_update", "slow_update", "imu", "fast_age", "slow_age", ...
    "esm_mode", "dt"];

modelPath = string(modelPath);
if isempty(net) || loadedModelPath ~= modelPath
    net = importNetworkFromONNX(modelPath, GenerateCoderFiles=false);
    loadedModelPath = modelPath;
end

networkInputNames = string(net.InputNames);
arguments = cell(1, numel(networkInputNames));
for index = 1:numel(networkInputNames)
    name = networkInputNames(index);
    if ~ismember(name, inputNames)
        error("Unexpected ONNX input: %s", name);
    end
    fieldName = char(name);
    if ~isfield(input, fieldName)
        error("Missing tracker input field: %s", name);
    end
    arguments{index} = dlarray(single(input.(fieldName)));
end

prediction = predict(net, arguments{:});
if isa(prediction, "dlarray")
    prediction = extractdata(prediction);
end
prediction = single(gather(prediction));
end
