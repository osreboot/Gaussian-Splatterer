#include "SplatPanelTools.h"

SplatPanelTools::SplatPanelTools(wxWindow *parent) : wxPanel(parent) {
    sizer = new wxBoxSizer(wxVERTICAL);

    sizerStaticInput = new wxStaticBoxSizer(wxVERTICAL, this, "Input Model Data");
    sizer->Add(sizerStaticInput, wxSizerFlags().Expand());

    sizerStaticTruth = new wxStaticBoxSizer(wxVERTICAL, this, "Build Truth Data");
    sizer->Add(sizerStaticTruth, wxSizerFlags().Expand());

    auto textCtrlCamerasCount = new wxStaticText(this, wxID_ANY, "Perspective Number");
    sizerStaticTruth->Add(textCtrlCamerasCount, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCtrlCamerasCount = new wxSpinCtrl(this);
    spinCtrlCamerasCount->SetRange(1, 128);
    spinCtrlCamerasCount->SetValue(8);
    spinCtrlCamerasCount->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &SplatPanelTools::onSpinCtrlCamerasCount, this);
    sizerStaticTruth->Add(spinCtrlCamerasCount, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    auto textCtrlCamerasDistance = new wxStaticText(this, wxID_ANY, "Perspective Distance");
    sizerStaticTruth->Add(textCtrlCamerasDistance, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCtrlCamerasDistance = new wxSpinCtrlDouble(this);
    spinCtrlCamerasDistance->SetRange(0.1, 10.0);
    spinCtrlCamerasDistance->SetDigits(2);
    spinCtrlCamerasDistance->SetIncrement(0.1);
    spinCtrlCamerasDistance->SetValue(2.0);
    spinCtrlCamerasDistance->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &SplatPanelTools::onSpinCtrlCamerasDistance, this);
    sizerStaticTruth->Add(spinCtrlCamerasDistance, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    sizerStaticTrain = new wxStaticBoxSizer(wxVERTICAL, this, "Train Splats");
    sizer->Add(sizerStaticTrain, wxSizerFlags().Expand());

    sizerStaticOutput = new wxStaticBoxSizer(wxVERTICAL, this, "Visualize Splats");
    sizer->Add(sizerStaticOutput, wxSizerFlags().Expand());

    SetSizerAndFit(sizer);
}

void SplatPanelTools::onSpinCtrlCamerasCount(wxSpinEvent& event) {
    splatCamerasUpdated = true;
    splatCamerasCount = event.GetValue();
}

void SplatPanelTools::onSpinCtrlCamerasDistance(wxSpinDoubleEvent& event) {
    splatCamerasUpdated = true;
    splatCamerasDistance = (float)event.GetValue();
}
