#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "ui/tools/UiPanelToolsTrain.h"
#include "ui/tools/UiPanelToolsTruth.h"

class UiPanelTools : public wxPanel {

private:
    wxBoxSizer* sizer;

    UiPanelToolsTruth* panelTruth;
    UiPanelToolsTrain* panelTrain;

    wxStaticBoxSizer* sizerStaticInput;
    wxStaticBoxSizer* sizerStaticOutput;

    wxCheckBox* checkBoxPreviewCamera;
    wxSpinCtrl* spinCtrlPreviewCamera;

public:
    UiPanelTools(wxWindow* parent);

    void onCheckBoxPreviewCamera(wxCommandEvent& event);
    void onSpinCtrlPreviewCamera(wxSpinEvent& event);

};
