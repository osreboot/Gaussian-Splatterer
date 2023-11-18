#pragma once

#include "UiFrame.h"
#include "FboRenderer.h"
#include "Project.h"

class UiPanelViewInput : public wxPanel {

private:
    UiFrame& getFrame() const;
    Project& getProject() const;

    wxBoxSizer* sizer;

    wxGLCanvas* canvas;
    FboRenderer* renderer;

    wxStaticText* textFrames;

public:
    wxGLContext* context;

    UiPanelViewInput(wxWindow *parent);
    ~UiPanelViewInput() override;

    void refreshProject();
    void refreshText();

    void render();

    void onPaint(wxPaintEvent& event);
    void onIdle(wxIdleEvent& event);

    DECLARE_EVENT_TABLE();

};
