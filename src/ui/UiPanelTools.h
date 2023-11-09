#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

class UiPanelTools : public wxPanel {

private:
    wxBoxSizer* sizer;

    wxStaticBoxSizer* sizerStaticInput;
    wxStaticBoxSizer* sizerStaticTruth;
    wxStaticBoxSizer* sizerStaticTrain;
    wxStaticBoxSizer* sizerStaticOutput;

    wxSpinCtrl* spinCtrlCamerasCount;
    wxSpinCtrlDouble* spinCtrlCamerasDistance;
    wxButton* buttonCamerasCapture;
    wxStaticText* textCamerasStatus;

    wxCheckBox* checkBoxPreviewCamera;
    wxSpinCtrl* spinCtrlPreviewCamera;

public:
    UiPanelTools(wxWindow *parent);

    void onSpinCtrlCamerasCount(wxSpinEvent& event);
    void onSpinCtrlCamerasDistance(wxSpinDoubleEvent& event);
    void onButtonCamerasCapture(wxCommandEvent& event);

    void UiPanelTools::onCheckBoxPreviewCamera(wxCommandEvent& event);
    void UiPanelTools::onSpinCtrlPreviewCamera(wxSpinEvent& event);

};
