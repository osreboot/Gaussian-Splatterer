#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "ui/UiFrame.h"
#include "Project.h"

class UiPanelParamsOther : public wxPanel{

private:
    UiFrame& frame;
    Project& getProject() const;

    wxSpinCtrlDouble* spinParamSizeMax;

    enum ParamIds {
        P_SIZE_MAX
    };

public:
    UiPanelParamsOther(wxWindow* parent, UiFrame& frame);

    void refreshProject();

    void onSpinParameter(wxSpinDoubleEvent& event);

};
