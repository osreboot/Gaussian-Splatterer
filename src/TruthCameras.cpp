#include "TruthCameras.h"

using namespace owl;

glm::vec3 TruthCameras::toGlmVec(const owl::vec3f& owlVec) {
    return {owlVec.x, owlVec.y, owlVec.z};
}

TruthCameras::TruthCameras() {
    refresh();
}

void TruthCameras::update(float delta) {
    previewTimer += delta;
}

Camera TruthCameras::getPreviewCamera() {
    previewPerspective = min(count - 1, max(-1, previewPerspective));

    return getCamera(previewPerspective);
}

Camera TruthCameras::getCamera(int index) const {
    assert(index >= -1 && index < count);

    static const vec3f target = {0.0f, 0.0f, 0.0f};

    if(index == -1) {
        return {vec3f(cos(previewTimer / 2.0f), 0.4f, sin(previewTimer / 2.0f)) * vec3f(10.0f),
                target, RENDER_FOV_X, RENDER_FOV_Y};
    } else return {locations[index], target, RENDER_FOV_X, RENDER_FOV_Y};
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
    bool out = updatedInput;
    updatedInput = false;
    return out;
}

void TruthCameras::refresh() {
    locations.clear();

    // Algorithm source for Fibonacci sphere placement: https://youtu.be/lctXaT9pxA0?si=xJQE1KXCH92s5tne&t=66
    float goldenRatio = (1.0f + sqrtf(5.0f)) / 2.0f;
    float angleStep = 2.0f * (float)M_PI * goldenRatio;
    for(int i = 0; i < count; i++) {
        float t = (float)i / (float)count;
        float angle1 = acosf(1.0f - 2.0f * t);
        float angle2 = angleStep * (float)i;

        locations.emplace_back(sinf(angle1) * cosf(angle2) * distance,
                               sinf(angle1) * sinf(angle2) * distance,
                               cosf(angle1) * distance);
    }

    updatedInput = true;
}
