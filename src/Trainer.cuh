#pragma once

#include "TruthCameras.h"
#include "ModelSplats.h"
#include "rtx/RtxHost.h"

class Trainer {

private:
    int lastCount = -1;

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

    float* varLocations = nullptr;
    float* avgGradLocations = nullptr;
    float* avgGradShs = nullptr;
    float* avgGradScales = nullptr;
    float* avgGradOpacities = nullptr;
    float* avgGradRotations = nullptr;

public:
    ModelSplats* model;

    std::vector<uint32_t*> truthFrameBuffersW;
    std::vector<uint32_t*> truthFrameBuffersB;
    std::vector<Camera> truthCameras;

    int iterations = 0;

    Trainer();
    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;
    Trainer(Trainer&&) = delete;
    Trainer& operator=(Trainer&&) = delete;

    ~Trainer();

    void render(uint32_t* frameBuffer, const Camera& camera);

    void captureTruths(const TruthCameras& cameras, RtxHost& rtx);

    void train(int iterations);
    void train(bool densify);

};
