#pragma once

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#define VERSION "1.0.0"

// Number of allowed training steps per second
#define AUTO_TRAIN_BUDGET 100.0f

// Resolution of truth images used by the training process
#define RENDER_RESOLUTION_X 1024
#define RENDER_RESOLUTION_Y 1024

// Maximum number of splats that a model can have (splats will automatically stop subdividing after reaching this limit)
#define SPLATS_LIMIT 1000000

#define SPLATS_SH_DEGREE 1
#define SPLATS_SH_COEF 4
