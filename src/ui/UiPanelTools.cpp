#include "UiPanelTools.h"

#include "UiFrame.h"

#include "Camera.h"
#include "Project.h"

Project& UiPanelTools::getProject() const {
    return *dynamic_cast<UiFrame*>(GetParent()->GetParent())->project;
}

UiPanelTools::UiPanelTools(wxWindow *parent) : wxPanel(parent) {
    sizer = new wxBoxSizer(wxHORIZONTAL);

    sizerStaticInput = new wxStaticBoxSizer(wxVERTICAL, this, "1. Input Model Data");
    sizer->Add(sizerStaticInput, wxSizerFlags().Expand().Border());

    panelTruth = new UiPanelToolsTruth(this);
    sizer->Add(panelTruth, wxSizerFlags().Expand().Border());

    panelTrain = new UiPanelToolsTrain(this);
    panelTrain->Disable();
    sizer->Add(panelTrain, wxSizerFlags().Expand().Border());

    sizerStaticOutput = new wxStaticBoxSizer(wxVERTICAL, this, "4. Visualize Splats");
    sizer->Add(sizerStaticOutput, wxSizerFlags().Expand().Border());

    checkBoxPreviewCamera = new wxCheckBox(this, wxID_ANY, "View Truth Perspective");
    checkBoxPreviewCamera->Bind(wxEVT_COMMAND_CHECKBOX_CLICKED, &UiPanelTools::onCheckBoxPreviewCamera, this);
    sizerStaticOutput->Add(checkBoxPreviewCamera, wxSizerFlags().Border(wxDOWN | wxUP));

    auto textCtrlPreviewCamera = new wxStaticText(this, wxID_ANY, "View Perspective Index");
    sizerStaticOutput->Add(textCtrlPreviewCamera, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCtrlPreviewCamera = new wxSpinCtrl(this);
    spinCtrlPreviewCamera->SetMinSize({64, -1});
    spinCtrlPreviewCamera->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelTools::onSpinCtrlPreviewCamera, this);
    spinCtrlPreviewCamera->Disable();
    sizerStaticOutput->Add(spinCtrlPreviewCamera, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    SetSizerAndFit(sizer);
}

void UiPanelTools::refreshProject() {
    checkBoxPreviewCamera->SetValue(getProject().previewIndex != -1);
    spinCtrlPreviewCamera->Enable(getProject().previewIndex != -1);
    spinCtrlPreviewCamera->SetValue(getProject().previewIndex + 1);
    refreshCameraCount();

    panelTruth->refreshProject();
    panelTrain->refreshProject();
}

void UiPanelTools::refreshCameraCount() {
    spinCtrlPreviewCamera->SetRange(1, Camera::getCamerasCount(getProject()));
    getProject().previewIndex = std::max(-1, std::min(Camera::getCamerasCount(getProject()) - 1, getProject().previewIndex));
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
