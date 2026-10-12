#include "mex.h"
#include "tracker.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define TRACKER_INPUT_COUNT 18

static const char *INPUT_NAMES[TRACKER_INPUT_COUNT] = {
    "radar", "eo", "esm", "radar_mask", "eo_mask", "esm_mask",
    "ref_pos", "rotmat_b2e", "uav_pos", "fast_link", "slow_link",
    "fast_update", "slow_update", "imu", "fast_age", "slow_age",
    "esm_mode", "dt"
};

static const size_t INPUT_COUNTS[TRACKER_INPUT_COUNT] = {
    1500, 1000, 1000, 500, 500, 500, 3, 4500, 1500,
    1500, 1500, 500, 500, 3000, 500, 500, 500, 1
};

static Tracker *create_tracker(const char *model_path) {
    Tracker *tracker;
#ifdef _WIN32
    {
        int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, model_path, -1, NULL, 0);
        wchar_t *wide_path;
        if (length == 0) {
            mexErrMsgIdAndTxt("coop_tracker:path", "Cannot convert model path to UTF-16.");
        }
        wide_path = (wchar_t *)mxMalloc((size_t)length * sizeof(wchar_t));
        if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, model_path, -1, wide_path, length) == 0) {
            mxFree(wide_path);
            mexErrMsgIdAndTxt("coop_tracker:path", "Cannot convert model path to UTF-16.");
        }
        tracker = tracker_create(wide_path);
        mxFree(wide_path);
    }
#else
    tracker = tracker_create(model_path);
#endif
    return tracker;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    const float *data[TRACKER_INPUT_COUNT];
    TrackerInputs inputs;
    Tracker *tracker;
    char *model_path;
    float *prediction;
    size_t index;

    if (nlhs > 1 || nrhs != TRACKER_INPUT_COUNT + 1) {
        mexErrMsgIdAndTxt(
            "coop_tracker:arguments",
            "Usage: prediction = coop_tracker_mex(modelPath, radar, ..., dt)");
    }
    model_path = mxArrayToUTF8String(prhs[0]);
    if (model_path == NULL) {
        mexErrMsgIdAndTxt("coop_tracker:path", "modelPath must be text.");
    }

    for (index = 0; index < TRACKER_INPUT_COUNT; ++index) {
        const mxArray *value = prhs[index + 1];
        if (!mxIsSingle(value) || mxIsComplex(value) ||
            mxGetNumberOfElements(value) != INPUT_COUNTS[index]) {
            mxFree(model_path);
            mexErrMsgIdAndTxt(
                "coop_tracker:input",
                "Input %s must be a real single array with %llu elements.",
                INPUT_NAMES[index], (unsigned long long)INPUT_COUNTS[index]);
        }
        data[index] = (const float *)mxGetData(value);
    }

    tracker = create_tracker(model_path);
    mxFree(model_path);
    if (tracker == NULL) {
        mexErrMsgIdAndTxt("coop_tracker:create", "Could not allocate the tracker.");
    }

    inputs = (TrackerInputs){
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
        data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], data[17]
    };
    plhs[0] = mxCreateNumericMatrix(TRACKER_OUTPUT_COUNT, 1, mxSINGLE_CLASS, mxREAL);
    prediction = (float *)mxGetData(plhs[0]);
    if (tracker_run(tracker, &inputs, prediction) != 0) {
        char error[512];
        strncpy(error, tracker_last_error(tracker), sizeof(error) - 1);
        error[sizeof(error) - 1] = '\0';
        tracker_destroy(tracker);
        mexErrMsgIdAndTxt("coop_tracker:inference", "%s", error);
    }
    tracker_destroy(tracker);
}
