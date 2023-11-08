#include "SplatPanelInput.h"
#include "SplatFrame.h"

using namespace std;

SplatPanelInput::SplatPanelInput(wxWindow *parent) : wxGLCanvas(parent) {
    context = new wxGLContext(this);

    wxGLCanvas::SetCurrent(*context);

    OWL_CUDA_CHECK(cudaMallocManaged(&frameBuffer, RENDER_RESOLUTION_X * RENDER_RESOLUTION_Y * sizeof(uint32_t)));
    glGenTextures(1, &textureFrameBuffer);
    glBindTexture(GL_TEXTURE_2D, textureFrameBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    OWL_CUDA_CHECK(cudaGraphicsGLRegisterImage(&textureCuda, textureFrameBuffer, GL_TEXTURE_2D, 0));

    rtx = new RtxHost({RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y});
    rtx->setSplatModel(R"(C:\Users\Calvin\Desktop\Archives\Development\Resources\Gecko 3d model\Gecko_m1.obj)",
                      R"(C:\Users\Calvin\Desktop\Archives\Development\Resources\Gecko 3d model\Files\Textures\Gecko Body Texture.BMP)");

    timeLastUpdate = chrono::high_resolution_clock::now();
}

SplatPanelInput::~SplatPanelInput() {
    delete context;
    delete rtx;
}

void SplatPanelInput::render() {
    wxGLCanvas::SetCurrent(*context);

    timeNow = chrono::high_resolution_clock::now();
    float delta = (float)chrono::duration_cast<chrono::nanoseconds>(timeNow - timeLastUpdate).count() / 1000000000.0f;
    //delta = min(delta, 0.2f);
    timeLastUpdate = timeNow;

    // Pre-update
    glClear(GL_COLOR_BUFFER_BIT);

    SplatFrame* frame = dynamic_cast<SplatFrame*>(GetParent()->GetParent());
    rtx->update(delta, (uint64_t)frameBuffer, *frame->truthCameras);

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

void SplatPanelInput::onPaint(wxPaintEvent& event) {
    if (!IsShown()) return;
    render();
}

void SplatPanelInput::onIdle(wxIdleEvent& event) {
    render();
    event.RequestMore();
}

BEGIN_EVENT_TABLE(SplatPanelInput, wxGLCanvas)
EVT_PAINT(SplatPanelInput::onPaint)
EVT_IDLE(SplatPanelInput::onIdle)
END_EVENT_TABLE()
