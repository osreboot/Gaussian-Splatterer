#pragma once

#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <vector>
#include <diff-gaussian-rasterization/third_party/glm/glm/glm.hpp>
#include <diff-gaussian-rasterization/third_party/glm/glm/gtc/quaternion.hpp>

#include "Config.h"

struct Camera {

    owl::vec3f location;
    owl::vec3f target;

    float degFovX, degFovY;

    glm::mat4 getView() const;
    glm::mat4 getProjection() const;

};

class TruthCameras {

public:
    static glm::vec3 toGlmVec(const owl::vec3f& owlVec);

private:
    bool updatedInput = false;

    int count = 32;
    float distance = 12.0f;
    float rotOffsetX = 0.0f, rotOffsetY = 0.0f;

public:
    int previewPerspective = -1;
    float previewTimer = 0.0f;

    std::vector<owl::vec3f> locations;

    TruthCameras();

    void update(float delta);

    Camera getPreviewCamera();
    Camera getCamera(int index) const;

    void setCount(int count);
    void setDistance(float distance);
    void setRotationOffset(float x, float y);

    int getCount() const { return count; }
    float getDistance() const { return distance; }
    float getRotationOffsetX() const { return rotOffsetX; }
    float getRotationOffsetY() const { return rotOffsetY; }

    bool pollInputUpdate();

private:
    void refresh();

};
