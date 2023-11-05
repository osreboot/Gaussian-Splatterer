#include "Window.h"

Window::Window(int widthArg, int heightArg, const std::string& titleArg) {
    size = owl::vec2i(widthArg, heightArg);

    // Create the window
    glfwInit();
    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window = glfwCreateWindow(widthArg, heightArg, titleArg.c_str(), nullptr, nullptr);
    glfwMakeContextCurrent(window);

    // Create the frame buffer and CUDA texture
    cudaMallocManaged(&frameBuffer, size.x * size.y * sizeof(uint32_t));
    glGenTextures(1, &textureFrameBuffer);
    glBindTexture(GL_TEXTURE_2D, textureFrameBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    cudaGraphicsGLRegisterImage(&textureCuda, textureFrameBuffer, GL_TEXTURE_2D, 0);
}

void Window::preUpdate() {
    glClear(GL_COLOR_BUFFER_BIT);
    glfwPollEvents();
}

void Window::postUpdate() {
    OWL_CUDA_CHECK(cudaGraphicsMapResources(1, &textureCuda));

    // Copy the CUDA texture to the frame buffer
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, textureCuda, 0, 0);
    cudaMemcpy2DToArray(array, 0, 0, reinterpret_cast<const void*>(frameBuffer),
                        size.x * sizeof(uint32_t), size.x * sizeof(uint32_t), size.y, cudaMemcpyDeviceToDevice);

    // Configure OpenGL for frame buffer rendering
    glDisable(GL_LIGHTING);
    glColor3f(1.0f, 1.0f, 1.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureFrameBuffer);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, size.x, size.y);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0f, (float)size.x, 0.0f, (float)size.y, -1.0f, 1.0f);

    // Draw the frame buffer
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex3f(0.0f, (float)size.y, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f((float)size.x, (float)size.y, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3f((float)size.x, 0.0f, 0.0f);
    glEnd();

    OWL_CUDA_CHECK(cudaGraphicsUnmapResources(1, &textureCuda));

    glfwSwapBuffers(window);
}

bool Window::exiting() {
    return glfwWindowShouldClose(window);
}

uint32_t* Window::getFrameBuffer() const {
    return frameBuffer;
}

owl::vec2i Window::getSize() const {
    return size;
}
