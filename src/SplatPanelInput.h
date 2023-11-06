#pragma once

#include <wx/wx.h>
#include <wx/glcanvas.h>

#include <owl/helper/cuda.h>
#include <cuda_gl_interop.h>

#include "rtx/RtxHost.h"

#define RENDER_RESOLUTION_X 1024
#define RENDER_RESOLUTION_Y 1024

class SplatPanelInput : public wxGLCanvas {

private:
    wxGLContext* context;

    std::chrono::high_resolution_clock::time_point timeLastUpdate, timeNow;

    RtxHost* rtx;

    GLuint textureFrameBuffer = {0};
    cudaGraphicsResource_t textureCuda = nullptr;
    uint32_t* frameBuffer = nullptr;

public:
    SplatPanelInput(wxWindow *parent);
    ~SplatPanelInput();

    void render();

    void onPaint(wxPaintEvent& event);
    void onIdle(wxIdleEvent& event);

    DECLARE_EVENT_TABLE();

};
