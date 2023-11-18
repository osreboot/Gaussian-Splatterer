#pragma once

#include "UiFrame.h"
#include "FboRenderer.h"
#include "Project.h"

class UiPanelViewOutput : public wxPanel {

private:
    UiFrame& getFrame() const;
    Project& getProject() const;

    wxBoxSizer* sizer;

    wxGLCanvas* canvas;
    FboRenderer* renderer;

    wxStaticText* textIterations;
    wxStaticText* textSplats;

public:
    wxGLContext* context;

    UiPanelViewOutput(wxWindow *parent, wxGLContext* context);
    ~UiPanelViewOutput() override;

    void refreshProject();
    void refreshText();

    void render();

    void onPaint(wxPaintEvent& event);
    void onIdle(wxIdleEvent& event);

    DECLARE_EVENT_TABLE();

};
