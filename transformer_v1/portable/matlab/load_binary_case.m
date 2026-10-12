function data = load_binary_case(caseDir)
%LOAD_BINARY_CASE Read the shared float32 golden case into a MATLAB struct.

caseDir = string(caseDir);
manifest = jsondecode(fileread(fullfile(caseDir, "manifest.json")));
names = string(fieldnames(manifest));
data = struct();

for index = 1:numel(names)
    name = names(index);
    item = manifest.(name);
    shape = double(item.shape(:).');
    filePath = fullfile(caseDir, string(item.file));
    fileId = fopen(filePath, "rb");
    if fileId < 0
        error("Cannot open binary tensor: %s", filePath);
    end
    cleanup = onCleanup(@() fclose(fileId));
    values = fread(fileId, prod(shape), "single=>single");
    clear cleanup
    if numel(values) ~= prod(shape)
        error("Tensor %s has %d values; expected %d", name, numel(values), prod(shape));
    end
    % Binary files use C row-major order. Reverse dimensions before reshape,
    % then reverse axes to recover the ONNX logical shape in MATLAB.
    % MATLAB reshape requires at least two dimensions, whereas ONNX permits
    % rank-1 tensors such as dt with shape [1]. Keep rank-1 data as a column
    % vector; run_tracker passes the single element to the ONNX input.
    if numel(shape) == 1
        value = reshape(values, shape(1), 1);
    else
        value = reshape(values, fliplr(shape));
        value = permute(value, numel(shape):-1:1);
    end
    data.(name) = value;
end
end
