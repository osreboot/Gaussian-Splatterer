#include "UiPanelToolsTrain.h"

#include "ui/UiPanelTools.h"

UiFrame& UiPanelToolsTrain::getFrame() const {
    return *dynamic_cast<UiFrame*>(GetParent()->GetParent()->GetParent());
}

Project& UiPanelToolsTrain::getProject() const {
    return *getFrame().project;
}

UiPanelToolsTrain::UiPanelToolsTrain(wxWindow* parent) : wxPanel(parent) {
    sizer = new wxStaticBoxSizer(wxHORIZONTAL, this, "3. Train Splats");

    sizerAuto = new wxBoxSizer(wxVERTICAL);
    sizer->Add(sizerAuto, wxSizerFlags().Border());

    sizerAuto->Add(new wxStaticText(this, wxID_ANY, "Auto Train"));
    buttonAutoStart = new wxButton(this, wxID_ANY, "Start");
    buttonAutoStart->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelToolsTrain::onButtonAutoStart, this);
    sizerAuto->Add(buttonAutoStart);
    buttonAutoStop = new wxButton(this, wxID_ANY, "Stop");
    buttonAutoStop->Disable();
    buttonAutoStop->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelToolsTrain::onButtonAutoStop, this);
    sizerAuto->Add(buttonAutoStop);

    SetSizerAndFit(sizer);
}

void UiPanelToolsTrain::refreshProject() {}

void UiPanelToolsTrain::onButtonAutoStart(wxCommandEvent &event) {
    getFrame().autoTraining = true;
    getFrame().panelTools->panelTruth->Disable();
    buttonAutoStart->Disable();
    buttonAutoStop->Enable();
}

void UiPanelToolsTrain::onButtonAutoStop(wxCommandEvent &event) {
    getFrame().autoTraining = false;
    getFrame().panelTools->panelTruth->Enable();
    buttonAutoStart->Enable();
    buttonAutoStop->Disable();
}
