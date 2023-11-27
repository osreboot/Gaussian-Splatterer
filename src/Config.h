#pragma once

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

// Resolution of truth images used by the training process
#define RENDER_RESOLUTION_X 1024
#define RENDER_RESOLUTION_Y 1024

// Maximum number of splats that a model can have (splats will automatically stop subdividing after reaching this limit)
#define SPLATS_LIMIT 1000000
