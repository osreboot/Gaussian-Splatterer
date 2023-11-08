#include "SplatPanelTools.h"
#include "SplatFrame.h"

using namespace std;

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
    spinCtrlCamerasCount->SetMinSize({64, -1});
    spinCtrlCamerasCount->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &SplatPanelTools::onSpinCtrlCamerasCount, this);
    sizerStaticTruth->Add(spinCtrlCamerasCount, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    auto textCtrlCamerasDistance = new wxStaticText(this, wxID_ANY, "Perspective Distance");
    sizerStaticTruth->Add(textCtrlCamerasDistance, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCtrlCamerasDistance = new wxSpinCtrlDouble(this);
    spinCtrlCamerasDistance->SetRange(0.1, 20.0);
    spinCtrlCamerasDistance->SetDigits(2);
    spinCtrlCamerasDistance->SetIncrement(0.1);
    spinCtrlCamerasDistance->SetValue(frame->truthCameras->getDistance());
    spinCtrlCamerasDistance->SetMinSize({64, -1});
    spinCtrlCamerasDistance->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &SplatPanelTools::onSpinCtrlCamerasDistance, this);
    sizerStaticTruth->Add(spinCtrlCamerasDistance, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    sizerStaticTrain = new wxStaticBoxSizer(wxVERTICAL, this, "Train Splats");
    sizer->Add(sizerStaticTrain, wxSizerFlags().Expand());

    sizerStaticOutput = new wxStaticBoxSizer(wxVERTICAL, this, "Visualize Splats");
    sizer->Add(sizerStaticOutput, wxSizerFlags().Expand());

    checkBoxPreviewCamera = new wxCheckBox(this, wxID_ANY, "View Truth Perspective");
    checkBoxPreviewCamera->Bind(wxEVT_COMMAND_CHECKBOX_CLICKED, &SplatPanelTools::onCheckBoxPreviewCamera, this);
    sizerStaticOutput->Add(checkBoxPreviewCamera, wxSizerFlags().Border(wxDOWN | wxUP));

    auto textCtrlPreviewCamera = new wxStaticText(this, wxID_ANY, "View Perspective Index");
    sizerStaticOutput->Add(textCtrlPreviewCamera, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCtrlPreviewCamera = new wxSpinCtrl(this);
    spinCtrlPreviewCamera->SetRange(1, frame->truthCameras->getCount());
    spinCtrlPreviewCamera->SetValue(1);
    spinCtrlPreviewCamera->SetMinSize({64, -1});
    spinCtrlPreviewCamera->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &SplatPanelTools::onSpinCtrlPreviewCamera, this);
    spinCtrlPreviewCamera->Disable();
    sizerStaticOutput->Add(spinCtrlPreviewCamera, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    SetSizerAndFit(sizer);
}

void SplatPanelTools::onSpinCtrlCamerasCount(wxSpinEvent& event) {
    SplatFrame* frame = dynamic_cast<SplatFrame*>(GetParent()->GetParent());
    frame->truthCameras->setCount(event.GetValue());
    spinCtrlPreviewCamera->SetRange(1, event.GetValue());
}

void SplatPanelTools::onSpinCtrlCamerasDistance(wxSpinDoubleEvent& event) {
    SplatFrame* frame = dynamic_cast<SplatFrame*>(GetParent()->GetParent());
    frame->truthCameras->setDistance((float)event.GetValue());
}

void SplatPanelTools::onCheckBoxPreviewCamera(wxCommandEvent& event) {
    SplatFrame* frame = dynamic_cast<SplatFrame*>(GetParent()->GetParent());
    if(event.IsChecked()) {
        frame->truthCameras->previewPerspective = spinCtrlPreviewCamera->GetValue() - 1;
        spinCtrlPreviewCamera->Enable();
    } else {
        frame->truthCameras->previewPerspective = -1;
        spinCtrlPreviewCamera->Disable();
    }
}

void SplatPanelTools::onSpinCtrlPreviewCamera(wxSpinEvent& event) {
    SplatFrame* frame = dynamic_cast<SplatFrame*>(GetParent()->GetParent());
    frame->truthCameras->previewPerspective = event.GetValue() - 1;
}
