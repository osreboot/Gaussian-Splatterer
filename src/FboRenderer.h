#pragma once

#include "OpenGLIncludes.h"

// Wraps and manages an integer-based frame buffer object, and handles rendering the object (assuming an OpenGL context
// is already present).
class FboRenderer {

private:
    const int width, height;

    GLuint textureGL; // OpenGL texture handle
    cudaGraphicsResource_t textureCuda = nullptr; // CUDA-mapped version of the OpenGL texture

public:
    uint32_t* frameBuffer = nullptr; // Device pointer to 32 bit integer-based frame buffer object

    FboRenderer(int width, int height);

    // Renders the contents of the FBO, assuming an OpenGL context is already present
    void render(int viewportWidth, int viewportHeight);

};
