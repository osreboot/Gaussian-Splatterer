#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "ui/UiFrame.h"
#include "Project.h"

class UiPanelParamsDensify : public wxPanel{

private:
    UiFrame& frame;
    Project& getProject() const;

    wxSpinCtrlDouble* spinParamCullOpacity;
    wxSpinCtrlDouble* spinParamCullSize;
    wxSpinCtrlDouble* spinParamDensifyVariance;
    //wxSpinCtrlDouble* spinParamDensifySize;
    wxSpinCtrlDouble* spinParamSplitSize;
    wxSpinCtrlDouble* spinParamSplitDistance;
    wxSpinCtrlDouble* spinParamSplitScale;
    wxSpinCtrlDouble* spinParamCloneDistance;

    enum ParamIds {
        P_CULL_OPACITY,
        P_CULL_SIZE,
        P_DENSIFY_VARIANCE,
        //P_DENSIFY_SIZE,
        P_SPLIT_SIZE,
        P_SPLIT_DISTANCE,
        P_SPLIT_SCALE,
        P_CLONE_DISTANCE
    };

public:
    UiPanelParamsDensify(wxWindow* parent, UiFrame& frame);

    void refreshProject();

    void onSpinParameter(wxSpinDoubleEvent& event);

};
