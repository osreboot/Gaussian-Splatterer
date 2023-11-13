#pragma once

class ModelSplatsHost;

class ModelSplatsDevice {

public:
    int capacity;
    int shDegree;
    int shCoeffs;

    int count = 0;

    // Note: these are GPU memory locations!
    float* devLocations = nullptr;
    float* devShs = nullptr;
    float* devScales = nullptr;
    float* devOpacities = nullptr;
    float* devRotations = nullptr;

    ModelSplatsDevice(const ModelSplatsDevice& device);
    explicit ModelSplatsDevice(const ModelSplatsHost& host);

    ModelSplatsDevice& operator=(const ModelSplatsDevice&) = delete;
    ModelSplatsDevice(ModelSplatsDevice&&) = delete;
    ModelSplatsDevice& operator=(ModelSplatsDevice&&) = delete;

    ~ModelSplatsDevice();

};
