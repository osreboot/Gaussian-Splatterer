#pragma once

#include <vector>
#include <diff-gaussian-rasterization/third_party/glm/glm/gtc/quaternion.hpp>

class ModelSplatsDevice;

class ModelSplatsHost {

public:
    int capacity;
    int shDegree;
    int shCoeffs;

    int count = 0;

    float* locations = nullptr;
    float* shs = nullptr;
    float* scales = nullptr;
    float* opacities = nullptr;
    float* rotations = nullptr;

    ModelSplatsHost(int capacity, int shDegree, int shCoeffs);
    explicit ModelSplatsHost(const ModelSplatsDevice& device);

    ModelSplatsHost(const ModelSplatsHost&) = delete;
    ModelSplatsHost& operator=(const ModelSplatsHost&) = delete;
    ModelSplatsHost(ModelSplatsHost&&) = delete;
    ModelSplatsHost& operator=(ModelSplatsHost&&) = delete;

    ~ModelSplatsHost();

    void pushBack(glm::vec3 location, std::vector<float> sh, glm::vec3 scale, float opacity, glm::quat rotation);
    void copy(int indexTo, int indexFrom);

};
