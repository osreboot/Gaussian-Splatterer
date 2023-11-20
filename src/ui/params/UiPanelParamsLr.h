#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "ui/UiFrame.h"
#include "Project.h"

class UiPanelParamsLr : public wxPanel{

private:
    UiFrame& getFrame() const;
    Project& getProject() const;

    wxStaticBoxSizer* sizer;

    wxSpinCtrlDouble* spinLrLocation;
    wxSpinCtrlDouble* spinLrSh;
    wxSpinCtrlDouble* spinLrScale;
    wxSpinCtrlDouble* spinLrOpacity;
    wxSpinCtrlDouble* spinLrRotation;

    enum ParamIds {
        LR_LOCATION,
        LR_SH,
        LR_SCALE,
        LR_OPACITY,
        LR_ROTATION
    };

public:
    UiPanelParamsLr(wxWindow* parent);

    void refreshProject();

    void onSpinParameter(wxSpinDoubleEvent& event);

};
