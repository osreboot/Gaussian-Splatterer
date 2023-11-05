#pragma once

#include <glfw/glfw3.h>
#include <owl/helper/cuda.h>
#include <cuda_gl_interop.h>

#include "Window.h"

class Window {

private:
    GLFWwindow* window;

    owl::vec2i size = {0};
    GLuint textureFrameBuffer = {0};
    cudaGraphicsResource_t textureCuda = nullptr;
    uint32_t* frameBuffer = nullptr;

public:
    Window(int widthArg, int heightArg, const std::string& titleArg);

    void preUpdate();
    void postUpdate();

    bool exiting();

    uint32_t* getFrameBuffer() const;
    owl::vec2i getSize() const;

};
