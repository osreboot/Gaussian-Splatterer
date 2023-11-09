#include "UiPanelOutput.h"
#include "UiFrame.h"

using namespace std;

UiPanelOutput::UiPanelOutput(wxWindow *parent, wxGLContext* context) : wxGLCanvas(parent), context(context) {
    wxGLCanvas::SetCurrent(*context);

    OWL_CUDA_CHECK(cudaMallocManaged(&frameBuffer, RENDER_RESOLUTION_X * RENDER_RESOLUTION_Y * sizeof(uint32_t)));
    glGenTextures(1, &textureFrameBuffer);
    glBindTexture(GL_TEXTURE_2D, textureFrameBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    OWL_CUDA_CHECK(cudaGraphicsGLRegisterImage(&textureCuda, textureFrameBuffer, GL_TEXTURE_2D, 0));
}

void UiPanelOutput::render() {
    wxGLCanvas::SetCurrent(*context);

    // Pre-update
    glClear(GL_COLOR_BUFFER_BIT);

    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->trainer->render(frameBuffer, frame->truthCameras->getPreviewCamera());

    // Post-update
    OWL_CUDA_CHECK(cudaGraphicsMapResources(1, &textureCuda));

    // Copy the CUDA texture to the frame buffer
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, textureCuda, 0, 0);
    cudaMemcpy2DToArray(array, 0, 0, reinterpret_cast<const void*>(frameBuffer),
                        RENDER_RESOLUTION_X * sizeof(uint32_t), RENDER_RESOLUTION_X * sizeof(uint32_t), RENDER_RESOLUTION_Y, cudaMemcpyDeviceToDevice);

    // Configure OpenGL for frame buffer rendering
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureFrameBuffer);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, GetSize().x, GetSize().y);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);

    // Draw the frame buffer
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(-1.0f, -1.0f, 0.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex3f(-1.0f, 1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(1.0f, 1.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3f(1.0f, -1.0f, 0.0f);
    glEnd();

    OWL_CUDA_CHECK(cudaGraphicsUnmapResources(1, &textureCuda));

    SwapBuffers();
}

void UiPanelOutput::onPaint(wxPaintEvent& event) {
    if (!IsShown()) return;
    render();
}

void UiPanelOutput::onIdle(wxIdleEvent& event) {
    render();
    event.RequestMore();
}

BEGIN_EVENT_TABLE(UiPanelOutput, wxGLCanvas)
EVT_PAINT(UiPanelOutput::onPaint)
EVT_IDLE(UiPanelOutput::onIdle)
END_EVENT_TABLE()
