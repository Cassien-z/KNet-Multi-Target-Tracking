function outputPath = build_mex()
%BUILD_MEX Compile the MATLAB entry point against ONNX Runtime.

matlabDir = fileparts(mfilename("fullpath"));
portableDir = fileparts(matlabDir);
cDir = fullfile(portableDir, "c");
ortDir = fullfile(portableDir, "third_party", "onnxruntime-win-x64-1.18.1");

if ~ispc
    error("The packaged ONNX Runtime library currently targets Windows x64.");
end
if isempty(mex.getCompilerConfigurations("C", "Selected"))
    error("No C compiler is selected. Run 'mex -setup C' first.");
end

mex("-R2018a", ...
    "-I" + fullfile(cDir, "include"), ...
    "-I" + fullfile(ortDir, "include"), ...
    fullfile(matlabDir, "tracker_mex.c"), ...
    fullfile(cDir, "src", "tracker.c"), ...
    fullfile(ortDir, "lib", "onnxruntime.lib"), ...
    "-output", "coop_tracker_mex", ...
    "-outdir", matlabDir);

copyfile(fullfile(ortDir, "lib", "onnxruntime.dll"), matlabDir, "f");
outputPath = fullfile(matlabDir, "coop_tracker_mex." + mexext);
fprintf("Built %s\n", outputPath);
end
