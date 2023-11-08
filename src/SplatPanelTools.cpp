#include "SplatPanelTools.h"
#include "SplatFrame.h"

SplatPanelTools::SplatPanelTools(wxWindow *parent) : wxPanel(parent) {
    SplatFrame* frame = dynamic_cast<SplatFrame*>(GetParent()->GetParent());

    sizer = new wxBoxSizer(wxVERTICAL);

    sizerStaticInput = new wxStaticBoxSizer(wxVERTICAL, this, "Input Model Data");
    sizer->Add(sizerStaticInput, wxSizerFlags().Expand());

    sizerStaticTruth = new wxStaticBoxSizer(wxVERTICAL, this, "Build Truth Data");
    sizer->Add(sizerStaticTruth, wxSizerFlags().Expand());

    auto textCtrlCamerasCount = new wxStaticText(this, wxID_ANY, "Perspective Number");
    sizerStaticTruth->Add(textCtrlCamerasCount, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCtrlCamerasCount = new wxSpinCtrl(this);
    spinCtrlCamerasCount->SetRange(1, 128);
    spinCtrlCamerasCount->SetValue(frame->truthCameras->getCount());
    spinCtrlCamerasCount->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &SplatPanelTools::onSpinCtrlCamerasCount, this);
    sizerStaticTruth->Add(spinCtrlCamerasCount, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    auto textCtrlCamerasDistance = new wxStaticText(this, wxID_ANY, "Perspective Distance");
    sizerStaticTruth->Add(textCtrlCamerasDistance, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCtrlCamerasDistance = new wxSpinCtrlDouble(this);
    spinCtrlCamerasDistance->SetRange(0.1, 10.0);
    spinCtrlCamerasDistance->SetDigits(2);
    spinCtrlCamerasDistance->SetIncrement(0.1);
    spinCtrlCamerasDistance->SetValue(frame->truthCameras->getDistance());
    spinCtrlCamerasDistance->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &SplatPanelTools::onSpinCtrlCamerasDistance, this);
    sizerStaticTruth->Add(spinCtrlCamerasDistance, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    sizerStaticTrain = new wxStaticBoxSizer(wxVERTICAL, this, "Train Splats");
    sizer->Add(sizerStaticTrain, wxSizerFlags().Expand());

    sizerStaticOutput = new wxStaticBoxSizer(wxVERTICAL, this, "Visualize Splats");
    sizer->Add(sizerStaticOutput, wxSizerFlags().Expand());

    SetSizerAndFit(sizer);
}

void SplatPanelTools::onSpinCtrlCamerasCount(wxSpinEvent& event) {
    SplatFrame* frame = dynamic_cast<SplatFrame*>(GetParent()->GetParent());
    frame->truthCameras->setCount(event.GetValue());
}

void SplatPanelTools::onSpinCtrlCamerasDistance(wxSpinDoubleEvent& event) {
    SplatFrame* frame = dynamic_cast<SplatFrame*>(GetParent()->GetParent());
    frame->truthCameras->setDistance((float)event.GetValue());
}
