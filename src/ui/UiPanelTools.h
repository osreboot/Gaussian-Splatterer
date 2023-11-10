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
    wxSpinCtrlDouble* spinCtrlCamerasRotX;
    wxSpinCtrlDouble* spinCtrlCamerasRotY;
    wxButton* buttonCamerasRotRandom;
    wxButton* buttonCamerasCapture;
    wxStaticText* textCamerasStatus;

    wxStaticText* textIterationCount;
    wxButton* buttonTrain;
    wxButton* buttonTrain10;
    wxButton* buttonTrain100;
    wxButton* buttonTrainDensify;
    wxStaticText* textSplatCount;

    wxCheckBox* checkBoxPreviewCamera;
    wxSpinCtrl* spinCtrlPreviewCamera;

    void updateIterationCount();
    void updateSplatCount();

public:
    UiPanelTools(wxWindow *parent);

    void onSpinCtrlCamerasCount(wxSpinEvent& event);
    void onSpinCtrlCamerasDistance(wxSpinDoubleEvent& event);
    void onSpinCtrlCamerasRotX(wxSpinDoubleEvent& event);
    void onSpinCtrlCamerasRotY(wxSpinDoubleEvent& event);
    void onButtonCamerasRotRandom(wxCommandEvent& event);
    void onButtonCamerasCapture(wxCommandEvent& event);

    void onButtonTrain(wxCommandEvent& event);
    void onButtonTrain10(wxCommandEvent& event);
    void onButtonTrain100(wxCommandEvent& event);
    void onButtonTrainDensify(wxCommandEvent& event);

    void onCheckBoxPreviewCamera(wxCommandEvent& event);
    void onSpinCtrlPreviewCamera(wxSpinEvent& event);

};
