#ifndef COOP_TRACKER_H
#define COOP_TRACKER_H

#include <stddef.h>

#ifdef _WIN32
#include <wchar.h>
typedef wchar_t tracker_path_char_t;
#else
typedef char tracker_path_char_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TRACKER_SEQUENCE_LENGTH 100
#define TRACKER_UAV_COUNT 5
#define TRACKER_STATE_DIM 6
#define TRACKER_OUTPUT_COUNT (TRACKER_SEQUENCE_LENGTH * TRACKER_STATE_DIM)

typedef struct Tracker Tracker;

typedef struct TrackerInputs {
    const float *radar;
    const float *eo;
    const float *esm;
    const float *radar_mask;
    const float *eo_mask;
    const float *esm_mask;
    const float *ref_pos;
    const float *rotmat_b2e;
    const float *uav_pos;
    const float *fast_link;
    const float *slow_link;
    const float *fast_update;
    const float *slow_update;
    const float *imu;
    const float *fast_age;
    const float *slow_age;
    const float *esm_mode;
    const float *dt;
} TrackerInputs;

Tracker *tracker_create(const tracker_path_char_t *model_path);
int tracker_run(Tracker *tracker, const TrackerInputs *inputs, float *prediction);
const char *tracker_last_error(const Tracker *tracker);
void tracker_destroy(Tracker *tracker);

#ifdef __cplusplus
}
#endif

#endif
