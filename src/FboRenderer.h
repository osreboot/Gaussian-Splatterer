#pragma once

#include "OpenGLIncludes.h"

class FboRenderer {

private:
    const int width, height;

    GLuint textureFrameBuffer = {0};
    cudaGraphicsResource_t textureCuda = nullptr;

public:
    uint32_t* frameBuffer = nullptr;

    FboRenderer(int width, int height);

    void render(int viewportWidth, int viewportHeight);

};
