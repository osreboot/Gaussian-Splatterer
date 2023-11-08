#pragma once

#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <vector>

struct Camera {

    owl::vec3f location;
    owl::vec3f target;

    float degFovX, degFovY;

};

class TruthCameras {

private:
    bool updatedInput = false;
    bool updatedOutput = false;

    int count = 8;
    float distance = 2.0f;

public:
    int previewPerspective = -1;
    float previewTimer = 0.0f;

    std::vector<owl::vec3f> locations;

    TruthCameras();

    void update(float delta);

    Camera getActiveCamera();

    void setCount(int count);
    void setDistance(float distance);

    int getCount() const { return count; }
    float getDistance() const { return distance; }

    bool pollInputUpdate();
    bool pollOutputUpdate();

private:
    void refresh();

};