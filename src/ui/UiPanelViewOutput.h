#pragma once

#include "UiFrame.h"
#include "FboRenderer.h"
#include "Project.h"

class UiPanelViewOutput : public wxPanel {

private:
    UiFrame& frame;
    Project& getProject() const;

    wxBoxSizer* sizer;

    wxGLCanvas* canvas;
    FboRenderer* renderer;

    wxStaticText* textIterations;
    wxStaticText* textSplats;

public:
    wxGLContext* context;

    UiPanelViewOutput(wxWindow *parent, UiFrame& frame, wxGLContext* context);
    ~UiPanelViewOutput() override;

    void refreshProject();
    void refreshText();

    void render();

    void onPaint(wxPaintEvent& event);
    void onIdle(wxIdleEvent& event);

    DECLARE_EVENT_TABLE();

};
