#pragma once

#include <string>
#include <nlohmann/json.hpp>

class Project {

public:
    std::string pathModel;
    std::string pathTexture;

    struct CameraSphere {
        int count = 16;
        float distance = 10.0f;
        float fovDeg = 60.0f;
        float rotX = 0.0f;
        float rotY = 0.0f;

        NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(CameraSphere, count, distance, fovDeg, rotX, rotY);
    } sphere1, sphere2;

    int iterations = 0;
    int intervalCapture = 50;
    int intervalDensify = 200;

    int previewIndex = -1;
    float previewTimer = 0.0f;

    int renderResX = 2048;
    int renderResY = 2048;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Project,
                                                pathModel, pathTexture,
                                                sphere1, sphere2, iterations, intervalCapture, intervalDensify,
                                                previewIndex, previewTimer, renderResX, renderResY);

};
