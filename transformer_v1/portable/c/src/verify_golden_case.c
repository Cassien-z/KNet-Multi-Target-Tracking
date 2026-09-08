#include "tracker.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

typedef struct TensorFile {
    const char *name;
    size_t count;
} TensorFile;

static const TensorFile INPUT_FILES[] = {
    {"radar",1500}, {"eo",1000}, {"esm",1000},
    {"radar_mask",500}, {"eo_mask",500}, {"esm_mask",500},
    {"ref_pos",3}, {"rotmat_b2e",4500}, {"uav_pos",1500},
    {"fast_link",1500}, {"slow_link",1500},
    {"fast_update",500}, {"slow_update",500}, {"imu",3000},
    {"fast_age",500}, {"slow_age",500}, {"esm_mode",500}, {"dt",1}
};

static int load_tensor(const char *case_dir, const TensorFile *spec, float **data) {
    char path[1024];
    FILE *file;
    snprintf(path, sizeof(path), "%s/%s.bin", case_dir, spec->name);
    file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }
    *data = (float *)malloc(spec->count * sizeof(float));
    if (*data == NULL || fread(*data, sizeof(float), spec->count, file) != spec->count) {
        fprintf(stderr, "Cannot read complete tensor %s\n", path);
        fclose(file);
        free(*data);
        *data = NULL;
        return -1;
    }
    fclose(file);
    return 0;
}

int main(int argc, char **argv) {
    float *input[18] = {0};
    float *expected = NULL;
    float prediction[TRACKER_OUTPUT_COUNT];
    TrackerInputs inputs;
    Tracker *tracker = NULL;
    TensorFile expected_spec = {"prediction", TRACKER_OUTPUT_COUNT};
    size_t index;
    int return_code = 1;
    double max_abs_error = 0.0;
    double max_position_delta = 0.0;
    double max_velocity_error = 0.0;

    if (argc != 3) {
        fprintf(stderr, "Usage: verify_golden_case MODEL.onnx GOLDEN_CASE_DIR\n");
        return 2;
    }
    for (index = 0; index < 18; ++index) {
        if (load_tensor(argv[2], &INPUT_FILES[index], &input[index])) goto cleanup;
    }
    if (load_tensor(argv[2], &expected_spec, &expected)) goto cleanup;

    inputs = (TrackerInputs){
        input[0],input[1],input[2],input[3],input[4],input[5],input[6],input[7],input[8],
        input[9],input[10],input[11],input[12],input[13],input[14],input[15],input[16],input[17]
    };

#ifdef _WIN32
    {
        wchar_t model_path[1024];
        if (!MultiByteToWideChar(CP_UTF8, 0, argv[1], -1, model_path, 1024)) {
            fprintf(stderr, "Cannot convert model path to UTF-16\n");
            goto cleanup;
        }
        tracker = tracker_create(model_path);
    }
#else
    tracker = tracker_create(argv[1]);
#endif
    if (tracker == NULL || tracker_run(tracker, &inputs, prediction)) {
        fprintf(stderr, "Inference failed: %s\n", tracker_last_error(tracker));
        goto cleanup;
    }

    for (index = 0; index < TRACKER_OUTPUT_COUNT; ++index) {
        double error = fabs((double)prediction[index] - expected[index]);
        if (error > max_abs_error) max_abs_error = error;
    }
    for (index = 0; index < TRACKER_SEQUENCE_LENGTH; ++index) {
        size_t offset = index * TRACKER_STATE_DIM;
        double dx = (double)prediction[offset] - expected[offset];
        double dy = (double)prediction[offset + 2] - expected[offset + 2];
        double dz = (double)prediction[offset + 4] - expected[offset + 4];
        double position_delta = sqrt(dx * dx + dy * dy + dz * dz);
        double velocity_error_1 = fabs((double)prediction[offset + 1] - expected[offset + 1]);
        double velocity_error_2 = fabs((double)prediction[offset + 3] - expected[offset + 3]);
        double velocity_error_3 = fabs((double)prediction[offset + 5] - expected[offset + 5]);
        if (position_delta > max_position_delta) max_position_delta = position_delta;
        if (velocity_error_1 > max_velocity_error) max_velocity_error = velocity_error_1;
        if (velocity_error_2 > max_velocity_error) max_velocity_error = velocity_error_2;
        if (velocity_error_3 > max_velocity_error) max_velocity_error = velocity_error_3;
    }
    printf("max_abs_error=%.9g\n", max_abs_error);
    printf("max_position_delta_m=%.9g\n", max_position_delta);
    printf("max_velocity_abs_error=%.9g\n", max_velocity_error);
    return_code = (max_position_delta <= 1.0 && max_velocity_error <= 0.2) ? 0 : 3;

cleanup:
    tracker_destroy(tracker);
    free(expected);
    for (index = 0; index < 18; ++index) free(input[index]);
    return return_code;
}
