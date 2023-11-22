#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "ui/params/UiPanelParamsLr.h"
#include "ui/params/UiPanelParamsDensify.h"
#include "ui/params/UiPanelParamsOther.h"

class UiPanelParams : public wxPanel {

private:
    Project& getProject() const;

    wxBoxSizer* sizer;

public:
    UiPanelParamsLr* panelLr;
    UiPanelParamsDensify* panelDensify;
    UiPanelParamsOther* panelOther;

    UiPanelParams(wxWindow* parent);

    void refreshProject();

};
