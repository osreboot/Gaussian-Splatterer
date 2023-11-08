#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

class SplatPanelTools : public wxPanel {

private:
    wxBoxSizer* sizer;

    wxStaticBoxSizer* sizerStaticInput;
    wxStaticBoxSizer* sizerStaticTruth;
    wxStaticBoxSizer* sizerStaticTrain;
    wxStaticBoxSizer* sizerStaticOutput;

    wxSpinCtrl* spinCtrlCamerasCount;
    wxSpinCtrlDouble* spinCtrlCamerasDistance;

    wxCheckBox* checkBoxPreviewCamera;
    wxSpinCtrl* spinCtrlPreviewCamera;

public:
    SplatPanelTools(wxWindow *parent);

    void onSpinCtrlCamerasCount(wxSpinEvent& event);
    void onSpinCtrlCamerasDistance(wxSpinDoubleEvent& event);

    void SplatPanelTools::onCheckBoxPreviewCamera(wxCommandEvent& event);
    void SplatPanelTools::onSpinCtrlPreviewCamera(wxSpinEvent& event);

};
