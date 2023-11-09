#pragma once

#include "TruthCameras.h"
#include "ModelSplats.h"
#include "rtx/RtxHost.h"

class Trainer {

private:
    float* devBackground;
    float* devMatView;
    float* devMatProjView;
    float* devCameraLocation;
    float* devRasterized;

public:
    ModelSplats* model;

    std::vector<uint32_t*> truths;

    Trainer();
    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;
    Trainer(Trainer&&) = delete;
    Trainer& operator=(Trainer&&) = delete;

    ~Trainer();

    void render(uint32_t* frameBuffer, const Camera& camera);

    void captureTruths(const TruthCameras& cameras, RtxHost& rtx);

};
