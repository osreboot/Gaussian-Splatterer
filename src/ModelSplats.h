#pragma once

#include <vector>
#include <diff-gaussian-rasterization/third_party/glm/glm/gtc/quaternion.hpp>

class ModelSplats {

public:
    const int capacity;
    const int shDegree;
    const int shCoeffs;

    int count = 0;

    float* locations;
    float* shs;
    float* scales;
    float* opacities;
    float* rotations;

    float* devLocations;
    float* devShs;
    float* devScales;
    float* devOpacities;
    float* devRotations;

    ModelSplats(int capacity, int shDegree, int shCoeffs);
    ModelSplats(const ModelSplats&) = delete;
    ModelSplats& operator=(const ModelSplats&) = delete;
    ModelSplats(ModelSplats&&) = delete;
    ModelSplats& operator=(ModelSplats&&) = delete;

    ~ModelSplats();

    void pushBack(glm::vec3 location, std::vector<float> sh, glm::vec3 scale, float opacity, glm::quat rotation);

    void deviceBuffer() const;

};
