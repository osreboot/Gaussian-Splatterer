#pragma once

#include <string>

class Project {

public:
    std::string pathModel;
    std::string pathTexture;

    struct {
        int count = 16;
        float distance = 10.0f;
        float fovDeg = 60.0f;
        float rotX = 0.0f;
        float rotY = 0.0f;
    } sphere1, sphere2;

    int previewIndex = -1;
    float previewTimer = 0.0f;

};
