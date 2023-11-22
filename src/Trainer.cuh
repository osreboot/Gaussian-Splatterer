#pragma once

#include <vector>

class ModelSplatsDevice;
class Camera;
class Project;
class RtxHost;

class Trainer {

private:
    int lastCount = -1;
    int lastTruthCount = -1;

    float* devBackground;
    float* devMatView;
    float* devMatProjView;
    float* devCameraLocation;
    float* devRasterized;

    float* devVarLocations;
    float* devAvgGradLocations;
    float* devAvgGradShs;
    float* devAvgGradScales;
    float* devAvgGradOpacities;
    float* devAvgGradRotations;

    float* devLossPixels;

    float* devGradLocations;
    float* devGradShs;
    float* devGradScales;
    float* devGradOpacities;
    float* devGradRotations;

    float* devGradMean2D;
    float* devGradConic;
    float* devGradColor;
    float* devGradCov3D;

public:
    ModelSplatsDevice* model;

    std::vector<uint32_t*> truthFrameBuffersW;
    std::vector<uint32_t*> truthFrameBuffersB;
    std::vector<Camera> truthCameras;

    Trainer();

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;
    Trainer(Trainer&&) = delete;
    Trainer& operator=(Trainer&&) = delete;

    ~Trainer();

    void render(uint32_t* frameBuffer, int sizeX, int sizeY, float splatScale, const Camera& camera);

    void captureTruths(const Project& project, RtxHost& rtx);

    void train(Project& project, bool densify);

};
