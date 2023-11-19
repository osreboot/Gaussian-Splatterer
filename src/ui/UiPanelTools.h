#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "ui/tools/UiPanelToolsTrain.h"
#include "ui/tools/UiPanelToolsTruth.h"
#include "ui/tools/UiPanelToolsView.h"

class UiPanelTools : public wxPanel {

private:
    Project& getProject() const;

    wxBoxSizer* sizer;

    wxStaticBoxSizer* sizerStaticInput;

public:
    UiPanelToolsTruth* panelTruth;
    UiPanelToolsTrain* panelTrain;
    UiPanelToolsView* panelView;

    UiPanelTools(wxWindow* parent);

    void refreshProject();

};
