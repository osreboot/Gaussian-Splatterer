#pragma once

#include <string>
#include <nlohmann/json.hpp>

class Project {

public:
    std::string pathModel;
    std::string pathTextureDiffuse;

    struct CameraSphere {
        int count = 16;
        float distance = 10.0f;
        float fovDeg = 60.0f;
        float rotX = 0.0f;
        float rotY = 0.0f;

        NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(CameraSphere, count, distance, fovDeg, rotX, rotY);
    } sphere1, sphere2;

    int rtSamples = 100;

    float lrLocation = 0.00005f;
    float lrSh = 0.0001f;
    float lrScale = 0.00002f;
    float lrOpacity = 0.0001f;
    float lrRotation = 0.000025f;

    float paramScaleMax = 0.3f;

    float paramCullOpacity = 0.005f;
    float paramCullSize = 0.004f;
    float paramDensifyVariance = 2.0f;
    //float paramDensifySize = 0.02f;
    float paramSplitSize = 0.04f;
    float paramSplitDistance = 1.5f;
    float paramSplitScale = 0.8f;
    float paramCloneDistance = 1.6f;

    int iterations = 0;
    int intervalCapture = 50;
    int intervalDensify = 200;

    float previewTimer = 0.0f;
    int previewRtSamples = 50;
    float previewSplatScale = 1.0f;

    bool previewTruth = false;
    int previewTruthIndex = 0;

    bool previewFreeOrbit = true;
    float previewFreeOrbitSpeed = 0.5f;
    float previewFreeDistance = 10.0f;
    float previewFreeFovDeg = 60.0f;
    float previewFreeRotX = 25.0f;
    float previewFreeRotY = 0.0f;

    int renderResX = 2048;
    int renderResY = 2048;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Project,
                                                pathModel, pathTextureDiffuse,
                                                sphere1, sphere2, rtSamples,
                                                lrLocation, lrSh, lrScale, lrOpacity, lrRotation,
                                                paramScaleMax, paramCullOpacity, paramCullSize, paramDensifyVariance, /*paramDensifySize,*/ paramSplitSize, paramSplitDistance, paramSplitScale, paramCloneDistance,
                                                iterations, intervalCapture, intervalDensify,
                                                previewTimer, previewRtSamples, previewSplatScale, previewTruth, previewTruthIndex,
                                                previewFreeOrbit, previewFreeOrbitSpeed, previewFreeDistance, previewFreeFovDeg, previewFreeRotX, previewFreeRotY,
                                                renderResX, renderResY);

};
