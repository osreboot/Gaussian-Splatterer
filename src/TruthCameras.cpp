#include "TruthCameras.h"

using namespace owl;

void TruthCameras::refresh() {
    locations.clear();

    // Algorithm source for Fibonacci sphere placement: https://youtu.be/lctXaT9pxA0?si=xJQE1KXCH92s5tne&t=66
    float goldenRatio = (1.0f + sqrtf(5.0f)) / 2.0f;
    float angleStep = 2.0f * (float)M_PI * goldenRatio;
    for(int i = 0; i < count; i++){
        float t = (float)i / (float)count;
        float angle1 = acosf(1.0f - 2.0f * t);
        float angle2 = angleStep * (float)i;

        locations.emplace_back(sinf(angle1) * cosf(angle2) * distance,
                               sinf(angle1) * sinf(angle2) * distance,
                               cosf(angle1) * distance);
    }

    dirtyInput = true;
    dirtyOutput = true;
}

void TruthCameras::setCount(int countArg) {
    count = countArg;
    refresh();
}

void TruthCameras::setDistance(float distanceArg) {
    distance = distanceArg;
    refresh();
}

bool TruthCameras::pollInputUpdate() {
    bool out = dirtyInput;
    dirtyInput = false;
    return out;
}

bool TruthCameras::pollOutputUpdate() {
    bool out = dirtyOutput;
    dirtyOutput = false;
    return out;
}
