#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "ui/UiFrame.h"
#include "Project.h"

class UiPanelToolsTrain : public wxPanel{

private:
    UiFrame& getFrame() const;
    Project& getProject() const;

    wxStaticBoxSizer* sizer;

    wxButton* buttonAutoStart;
    wxButton* buttonAutoStop;

    wxSpinCtrlDouble* spinLrLocation;
    wxSpinCtrlDouble* spinLrSh;
    wxSpinCtrlDouble* spinLrScale;
    wxSpinCtrlDouble* spinLrOpacity;
    wxSpinCtrlDouble* spinLrRotation;

    wxStaticText* textIntervalCapture;
    wxStaticText* textIntervalDensify;
    wxSpinCtrl* spinIntervalCapture;
    wxSpinCtrl* spinIntervalDensify;

    enum ParameterIds {
        LR_LOCATION,
        LR_SH,
        LR_SCALE,
        LR_OPACITY,
        LR_ROTATION
    };

public:
    UiPanelToolsTrain(wxWindow* parent);

    void refreshProject();
    void refreshText();

    void onButtonAutoStart(wxCommandEvent& event);
    void onButtonAutoStop(wxCommandEvent& event);

    void onSpinParameter(wxSpinDoubleEvent& event);

    void onSpinIntervalCapture(wxSpinEvent& event);
    void onSpinIntervalDensify(wxSpinEvent& event);

};
