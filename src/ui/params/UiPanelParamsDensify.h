#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "ui/UiFrame.h"
#include "Project.h"

class UiPanelParamsDensify : public wxPanel{

private:
    UiFrame& getFrame() const;
    Project& getProject() const;

    wxStaticBoxSizer* sizer;

    wxSpinCtrlDouble* spinParamCullOpacity;
    wxSpinCtrlDouble* spinParamCullSize;
    wxSpinCtrlDouble* spinParamSplitVariance;
    wxSpinCtrlDouble* spinParamSplitSize;
    wxSpinCtrlDouble* spinParamSplitDistance;
    wxSpinCtrlDouble* spinParamSplitScale;
    wxSpinCtrlDouble* spinParamCloneDistance;

    enum ParamIds {
        P_CULL_OPACITY,
        P_CULL_SIZE,
        P_SPLIT_VARIANCE,
        P_SPLIT_SIZE,
        P_SPLIT_DISTANCE,
        P_SPLIT_SCALE,
        P_CLONE_DISTANCE
    };

public:
    UiPanelParamsDensify(wxWindow* parent);

    void refreshProject();

    void onSpinParameter(wxSpinDoubleEvent& event);

};
