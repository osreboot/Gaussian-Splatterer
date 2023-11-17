#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "ui/tools/UiPanelToolsTrain.h"
#include "ui/tools/UiPanelToolsTruth.h"

class UiPanelTools : public wxPanel {

private:
    Project& getProject() const;

    wxBoxSizer* sizer;

    wxStaticBoxSizer* sizerStaticInput;
    wxStaticBoxSizer* sizerStaticOutput;

    wxCheckBox* checkBoxPreviewCamera;
    wxSpinCtrl* spinCtrlPreviewCamera;

public:
    UiPanelToolsTruth* panelTruth;
    UiPanelToolsTrain* panelTrain;

    UiPanelTools(wxWindow* parent);

    void refreshCameraCount();

    void onCheckBoxPreviewCamera(wxCommandEvent& event);
    void onSpinCtrlPreviewCamera(wxSpinEvent& event);

};
