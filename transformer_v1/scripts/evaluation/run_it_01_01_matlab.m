function run_it_01_01_matlab(outputPath)
%RUN_IT_01_01_MATLAB Execute MATLAB checks for test case IT-01-01.

projectRoot = pwd;
addpath(fullfile(projectRoot, "portable", "matlab"));
modelPath = fullfile(projectRoot, "portable", "artifacts", "coop_tracker_v7.onnx");
caseDir = fullfile(projectRoot, "portable", "artifacts", "golden_case");

golden = verify_golden_case(modelPath, caseDir);
data = load_binary_case(caseDir);
data = rmfield(data, "prediction");
repeatOutputs = cell(1, 3);
for index = 1:3
    repeatOutputs{index} = run_tracker(modelPath, data);
end
repeatMaxAbsDifference = max(abs(double(repeatOutputs{2}) - double(repeatOutputs{1})), [], "all");
repeatMaxAbsDifference = max(repeatMaxAbsDifference, ...
    max(abs(double(repeatOutputs{3}) - double(repeatOutputs{1})), [], "all"));

missingFieldRejected = false;
missingFieldMessage = "";
try
    invalid = rmfield(data, "radar");
    run_tracker(modelPath, invalid);
catch exception
    missingFieldMessage = string(exception.message);
    missingFieldRejected = contains(missingFieldMessage, "Missing tracker input field: radar");
end

wrongCountRejected = false;
wrongCountMessage = "";
try
    invalid = data;
    invalid.radar = invalid.radar(1:end-1);
    run_tracker(modelPath, invalid);
catch exception
    wrongCountMessage = string(exception.message);
    wrongCountRejected = contains(wrongCountMessage, "expected 1500");
end

result = struct( ...
    "golden_case", golden, ...
    "repeat_max_abs_difference", repeatMaxAbsDifference, ...
    "missing_field_rejected", missingFieldRejected, ...
    "missing_field_message", missingFieldMessage, ...
    "wrong_element_count_rejected", wrongCountRejected, ...
    "wrong_element_count_message", wrongCountMessage);

fileId = fopen(outputPath, "w", "n", "UTF-8");
if fileId < 0
    error("Cannot create MATLAB result file: %s", outputPath);
end
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, "%s\n", jsonencode(result, PrettyPrint=true));

if ~missingFieldRejected || ~wrongCountRejected
    error("MATLAB negative-input checks did not produce the expected errors.");
end
end
