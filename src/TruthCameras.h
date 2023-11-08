#pragma once

#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <vector>

class TruthCameras {

private:
    bool dirtyInput = false;
    bool dirtyOutput = false;

    int count = 8;
    float distance = 2.0f;

    void refresh();

public:
    std::vector<owl::vec3f> locations;

    void setCount(int count);
    void setDistance(float distance);

    int getCount() const { return count; }
    float getDistance() const { return distance; }

    bool pollInputUpdate();
    bool pollOutputUpdate();

};
