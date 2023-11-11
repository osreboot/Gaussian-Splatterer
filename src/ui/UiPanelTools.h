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
    wxStaticText* textSplatCount;
    wxButton* buttonTrain;
    wxButton* buttonTrain10;
    wxButton* buttonTrain100;
    wxButton* buttonTrainDensify;
    wxButton* buttonTrainAutoStart;
    wxButton* buttonTrainAutoStop;

    wxCheckBox* checkBoxPreviewCamera;
    wxSpinCtrl* spinCtrlPreviewCamera;

public:
    UiPanelTools(wxWindow *parent);

    void updateIterationCount();
    void updateSplatCount();

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
    void onButtonTrainAutoStart(wxCommandEvent& event);
    void onButtonTrainAutoStop(wxCommandEvent& event);

    void onCheckBoxPreviewCamera(wxCommandEvent& event);
    void onSpinCtrlPreviewCamera(wxSpinEvent& event);

};
