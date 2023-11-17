#pragma once

#include "OpenGLIncludes.h"

class UiPanelViewInput : public wxGLCanvas {

private:
    GLuint textureFrameBuffer = {0};
    cudaGraphicsResource_t textureCuda = nullptr;
    uint32_t* frameBuffer = nullptr;

public:
    wxGLContext* context;

    UiPanelViewInput(wxWindow *parent);
    ~UiPanelViewInput() override;

    void render();

    void onPaint(wxPaintEvent& event);
    void onIdle(wxIdleEvent& event);

    DECLARE_EVENT_TABLE();

};
