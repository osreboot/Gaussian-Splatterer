#pragma once

#include "TruthCameras.h"
#include "ModelSplats.h"

class Trainer {

private:
    float* devBackground;
    float* devMatView;
    float* devMatProjView;
    float* devCameraLocation;
    float* devRasterized;

public:
    ModelSplats* model;

    Trainer();
    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;
    Trainer(Trainer&&) = delete;
    Trainer& operator=(Trainer&&) = delete;

    ~Trainer();

    void render(uint32_t* frameBuffer, TruthCameras& cameras);

};
