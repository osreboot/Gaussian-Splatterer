#include "UiPanelTools.h"

#include "UiFrame.h"

#include "Camera.h"
#include "Project.h"

using namespace std;

UiPanelTools::UiPanelTools(wxWindow *parent) : wxPanel(parent) {
    Project& project = *dynamic_cast<UiFrame*>(GetParent()->GetParent())->project;

    sizer = new wxBoxSizer(wxHORIZONTAL);

    sizerStaticInput = new wxStaticBoxSizer(wxVERTICAL, this, "1. Input Model Data");
    sizer->Add(sizerStaticInput, wxSizerFlags().Expand().Border());

    panelTruth = new UiPanelToolsTruth(this);
    sizer->Add(panelTruth, wxSizerFlags().Expand().Border());

    panelTrain = new UiPanelToolsTrain(this);
    sizer->Add(panelTrain, wxSizerFlags().Expand().Border());

    sizerStaticOutput = new wxStaticBoxSizer(wxVERTICAL, this, "4. Visualize Splats");
    sizer->Add(sizerStaticOutput, wxSizerFlags().Expand().Border());

    checkBoxPreviewCamera = new wxCheckBox(this, wxID_ANY, "View Truth Perspective");
    checkBoxPreviewCamera->Bind(wxEVT_COMMAND_CHECKBOX_CLICKED, &UiPanelTools::onCheckBoxPreviewCamera, this);
    sizerStaticOutput->Add(checkBoxPreviewCamera, wxSizerFlags().Border(wxDOWN | wxUP));

    auto textCtrlPreviewCamera = new wxStaticText(this, wxID_ANY, "View Perspective Index");
    sizerStaticOutput->Add(textCtrlPreviewCamera, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCtrlPreviewCamera = new wxSpinCtrl(this);
    spinCtrlPreviewCamera->SetRange(1, Camera::getCamerasCount(project));
    spinCtrlPreviewCamera->SetValue(1);
    spinCtrlPreviewCamera->SetMinSize({64, -1});
    spinCtrlPreviewCamera->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelTools::onSpinCtrlPreviewCamera, this);
    spinCtrlPreviewCamera->Disable();
    sizerStaticOutput->Add(spinCtrlPreviewCamera, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    SetSizerAndFit(sizer);
}

void UiPanelTools::onCheckBoxPreviewCamera(wxCommandEvent& event) {
    Project& project = *dynamic_cast<UiFrame*>(GetParent()->GetParent())->project;
    if(event.IsChecked()) {
        project.previewIndex = spinCtrlPreviewCamera->GetValue() - 1;
        spinCtrlPreviewCamera->Enable();
    } else {
        project.previewIndex = -1;
        spinCtrlPreviewCamera->Disable();
    }
}

void UiPanelTools::onSpinCtrlPreviewCamera(wxSpinEvent& event) {
    Project& project = *dynamic_cast<UiFrame*>(GetParent()->GetParent())->project;
    project.previewIndex = event.GetValue() - 1;
}
