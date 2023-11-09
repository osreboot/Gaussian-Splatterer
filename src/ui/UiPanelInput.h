#pragma once

#include <wx/wx.h>
#include <wx/glcanvas.h>

#include <owl/helper/cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Config.h"

class UiPanelInput : public wxGLCanvas {

private:
    GLuint textureFrameBuffer = {0};
    cudaGraphicsResource_t textureCuda = nullptr;
    uint32_t* frameBuffer = nullptr;

public:
    wxGLContext* context;

    UiPanelInput(wxWindow *parent);
    ~UiPanelInput() override;

    void render();

    void onPaint(wxPaintEvent& event);
    void onIdle(wxIdleEvent& event);

    DECLARE_EVENT_TABLE();

};
