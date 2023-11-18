#pragma once

#include <wx/wx.h>

#include "ui/UiFrame.h"
#include "Project.h"

class UiPanelToolsTrain : public wxPanel{

private:
    UiFrame& getFrame() const;
    Project& getProject() const;

    wxStaticBoxSizer* sizer;

    wxButton* buttonAutoStart;
    wxButton* buttonAutoStop;

    wxStaticText* textIntervalCapture;
    wxStaticText* textIntervalDensify;
    wxSpinCtrl* spinIntervalCapture;
    wxSpinCtrl* spinIntervalDensify;

public:
    UiPanelToolsTrain(wxWindow* parent);

    void refreshProject();
    void refreshText();

    void onButtonAutoStart(wxCommandEvent& event);
    void onButtonAutoStop(wxCommandEvent& event);

    void onSpinIntervalCapture(wxSpinEvent& event);
    void onSpinIntervalDensify(wxSpinEvent& event);

};
