#pragma once

#include <vector>

class ModelSplatsDevice;
class Camera;
class Project;
class RtxHost;

class Trainer {

private:
    // Variables to track how many splats/truth frames were present last iteration. If these don't match the current
    // values, we need to reallocate all host and device buffers.
    int lastCount = -1;
    int lastTruthCount = -1;

    // Device training data (used between all models/projects)
    float* devBackground;
    float* devMatView;
    float* devMatProjView;
    float* devCameraLocation;
    float* devRasterized;

    // Device averaged/accumulated training data (used across all truth frames for a single training iteration)
    float* devVarLocations;
    float* devAvgGradLocations;
    float* devAvgGradShs;
    float* devAvgGradScales;
    float* devAvgGradOpacities;
    float* devAvgGradRotations;

    // Device per-pixel loss for a single truth frame
    float* devLossPixels;

    // Device per-frame training data
    float* devGradLocations;
    float* devGradShs;
    float* devGradScales;
    float* devGradOpacities;
    float* devGradRotations;

    // Device data used by the backward rasterizer
    float* devGradMean2D;
    float* devGradConic;
    float* devGradColor;
    float* devGradCov3D;

public:
    ModelSplatsDevice* model;

    std::vector<uint32_t*> truthFrameBuffersW; // Truth FBO pointers (with white backgrounds)
    std::vector<uint32_t*> truthFrameBuffersB; // Truth FBO pointers (with black backgrounds)
    std::vector<Camera> truthCameras;

    Trainer();

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;
    Trainer(Trainer&&) = delete;
    Trainer& operator=(Trainer&&) = delete;

    ~Trainer();

    // Render the current model to a given FBO. Size parameters represent the FBO resolution. Uses the given camera and
    // the debug scale modifier for splats.
    void render(uint32_t* frameBuffer, int sizeX, int sizeY, float splatScale, const Camera& camera);

    // Capture truth frames using the given project settings and ray tracer
    void captureTruths(const Project& project, RtxHost& rtx);

    // Advance one training iteration (assumes model is loaded and truth data is present)
    void train(Project& project, bool densify);

};
