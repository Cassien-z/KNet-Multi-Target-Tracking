function result = verify_golden_case(modelPath, caseDir)
%VERIFY_GOLDEN_CASE Compare MATLAB inference with the frozen Python output.

data = load_binary_case(caseDir);
expected = data.prediction;
data = rmfield(data, "prediction");
actual = run_tracker(modelPath, data);

delta = abs(double(actual) - double(expected));
positionDelta = sqrt(sum((double(actual(:,:,[1 3 5])) - ...
    double(expected(:,:,[1 3 5]))).^2, 3));
velocityDelta = delta(:,:,[2 4 6]);

result = struct( ...
    "max_abs_error", max(delta, [], "all"), ...
    "mean_abs_error", mean(delta, "all"), ...
    "max_position_delta_m", max(positionDelta, [], "all"), ...
    "mean_position_delta_m", mean(positionDelta, "all"), ...
    "max_velocity_abs_error", max(velocityDelta, [], "all"));

disp(result)
if result.max_position_delta_m > 1.0 || result.max_velocity_abs_error > 0.2
    error("MATLAB and Python golden outputs are outside the acceptance tolerance.");
end
end
