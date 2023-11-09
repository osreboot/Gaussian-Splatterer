#pragma once

#include "Config.h"
#include "UiPanelInput.h"

class UiPanelOutput : public wxGLCanvas {

private:
    GLuint textureFrameBuffer = {0};
    cudaGraphicsResource_t textureCuda = nullptr;
    uint32_t* frameBuffer = nullptr;

public:
    wxGLContext* context;

    UiPanelOutput(wxWindow *parent, wxGLContext* context);

    void render();

    void onPaint(wxPaintEvent& event);
    void onIdle(wxIdleEvent& event);

    DECLARE_EVENT_TABLE();

};
