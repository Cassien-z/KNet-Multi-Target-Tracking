#include "tracker.h"

#include <onnxruntime_c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRACKER_INPUT_COUNT 18

struct Tracker {
    const OrtApi *api;
    OrtEnv *env;
    OrtSessionOptions *options;
    OrtSession *session;
    OrtMemoryInfo *memory_info;
    char error[512];
};

static const char *INPUT_NAMES[TRACKER_INPUT_COUNT] = {
    "radar", "eo", "esm", "radar_mask", "eo_mask", "esm_mask",
    "ref_pos", "rotmat_b2e", "uav_pos", "fast_link", "slow_link",
    "fast_update", "slow_update", "imu", "fast_age", "slow_age",
    "esm_mode", "dt"
};
static const char *OUTPUT_NAMES[1] = {"prediction"};

static const int64_t SHAPE_RADAR[] = {1, 100, 5, 3};
static const int64_t SHAPE_ANGLE[] = {1, 100, 5, 2};
static const int64_t SHAPE_MASK[] = {1, 100, 5};
static const int64_t SHAPE_REF[] = {1, 3};
static const int64_t SHAPE_ROT[] = {1, 100, 5, 3, 3};
static const int64_t SHAPE_XYZ[] = {1, 100, 5, 3};
static const int64_t SHAPE_IMU[] = {1, 100, 5, 6};
static const int64_t SHAPE_DT[] = {1};

static int set_status(Tracker *tracker, OrtStatus *status) {
    if (status == NULL) return 0;
    snprintf(tracker->error, sizeof(tracker->error), "%s", tracker->api->GetErrorMessage(status));
    tracker->api->ReleaseStatus(status);
    return -1;
}

static int make_tensor(
    Tracker *tracker,
    float *data,
    size_t count,
    const int64_t *shape,
    size_t rank,
    OrtValue **value
) {
    return set_status(tracker, tracker->api->CreateTensorWithDataAsOrtValue(
        tracker->memory_info, data, count * sizeof(float), shape, rank,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, value));
}

Tracker *tracker_create(const tracker_path_char_t *model_path) {
    Tracker *tracker = (Tracker *)calloc(1, sizeof(Tracker));
    if (tracker == NULL) return NULL;
    tracker->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (tracker->api == NULL) {
        snprintf(tracker->error, sizeof(tracker->error), "Cannot obtain ONNX Runtime API");
        return tracker;
    }
    if (set_status(tracker, tracker->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "coop_tracker", &tracker->env)) ||
        set_status(tracker, tracker->api->CreateSessionOptions(&tracker->options)) ||
        set_status(tracker, tracker->api->SetIntraOpNumThreads(tracker->options, 1)) ||
        set_status(tracker, tracker->api->SetSessionGraphOptimizationLevel(tracker->options, ORT_ENABLE_ALL)) ||
        set_status(tracker, tracker->api->CreateSession(tracker->env, model_path, tracker->options, &tracker->session)) ||
        set_status(tracker, tracker->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &tracker->memory_info))) {
        return tracker;
    }
    return tracker;
}

int tracker_run(Tracker *tracker, const TrackerInputs *inputs, float *prediction) {
    OrtValue *values[TRACKER_INPUT_COUNT] = {0};
    OrtValue *output = NULL;
    float *output_data = NULL;
    int result = -1;
    size_t index;
    float *data[TRACKER_INPUT_COUNT];

    if (tracker == NULL || tracker->session == NULL || inputs == NULL || prediction == NULL) return -1;
    tracker->error[0] = '\0';
    data[0]=(float *)inputs->radar; data[1]=(float *)inputs->eo; data[2]=(float *)inputs->esm;
    data[3]=(float *)inputs->radar_mask; data[4]=(float *)inputs->eo_mask; data[5]=(float *)inputs->esm_mask;
    data[6]=(float *)inputs->ref_pos; data[7]=(float *)inputs->rotmat_b2e; data[8]=(float *)inputs->uav_pos;
    data[9]=(float *)inputs->fast_link; data[10]=(float *)inputs->slow_link;
    data[11]=(float *)inputs->fast_update; data[12]=(float *)inputs->slow_update; data[13]=(float *)inputs->imu;
    data[14]=(float *)inputs->fast_age; data[15]=(float *)inputs->slow_age; data[16]=(float *)inputs->esm_mode;
    data[17]=(float *)inputs->dt;

#define MAKE(I, COUNT, SHAPE) if (make_tensor(tracker, data[I], COUNT, SHAPE, sizeof(SHAPE)/sizeof((SHAPE)[0]), &values[I])) goto cleanup
    MAKE(0,1500,SHAPE_RADAR); MAKE(1,1000,SHAPE_ANGLE); MAKE(2,1000,SHAPE_ANGLE);
    MAKE(3,500,SHAPE_MASK); MAKE(4,500,SHAPE_MASK); MAKE(5,500,SHAPE_MASK);
    MAKE(6,3,SHAPE_REF); MAKE(7,4500,SHAPE_ROT); MAKE(8,1500,SHAPE_XYZ);
    MAKE(9,1500,SHAPE_XYZ); MAKE(10,1500,SHAPE_XYZ);
    MAKE(11,500,SHAPE_MASK); MAKE(12,500,SHAPE_MASK); MAKE(13,3000,SHAPE_IMU);
    MAKE(14,500,SHAPE_MASK); MAKE(15,500,SHAPE_MASK); MAKE(16,500,SHAPE_MASK);
    MAKE(17,1,SHAPE_DT);
#undef MAKE

    if (set_status(tracker, tracker->api->Run(
        tracker->session, NULL, INPUT_NAMES, (const OrtValue *const *)values,
        TRACKER_INPUT_COUNT, OUTPUT_NAMES, 1, &output))) goto cleanup;
    if (set_status(tracker, tracker->api->GetTensorMutableData(output, (void **)&output_data))) goto cleanup;
    memcpy(prediction, output_data, TRACKER_OUTPUT_COUNT * sizeof(float));
    result = 0;

cleanup:
    if (output != NULL) tracker->api->ReleaseValue(output);
    for (index = 0; index < TRACKER_INPUT_COUNT; ++index) {
        if (values[index] != NULL) tracker->api->ReleaseValue(values[index]);
    }
    return result;
}

const char *tracker_last_error(const Tracker *tracker) {
    return tracker == NULL ? "Tracker allocation failed" : tracker->error;
}

void tracker_destroy(Tracker *tracker) {
    if (tracker == NULL) return;
    if (tracker->api != NULL) {
        if (tracker->memory_info != NULL) tracker->api->ReleaseMemoryInfo(tracker->memory_info);
        if (tracker->session != NULL) tracker->api->ReleaseSession(tracker->session);
        if (tracker->options != NULL) tracker->api->ReleaseSessionOptions(tracker->options);
        if (tracker->env != NULL) tracker->api->ReleaseEnv(tracker->env);
    }
    free(tracker);
}
