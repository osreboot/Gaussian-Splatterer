#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "ui/params/UiPanelParamsLr.h"

class UiPanelParams : public wxPanel {

private:
    Project& getProject() const;

    wxBoxSizer* sizer;

public:
    UiPanelParamsLr* panelLr;

    UiPanelParams(wxWindow* parent);

    void refreshProject();

};
