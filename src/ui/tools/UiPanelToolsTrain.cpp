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

    wxBoxSizer* sizerAuto = new wxBoxSizer(wxVERTICAL);
    sizer->Add(sizerAuto, wxSizerFlags().Border());

    sizerAuto->Add(new wxStaticText(this, wxID_ANY, "Auto Train"));
    buttonAutoStart = new wxButton(this, wxID_ANY, "Start");
    buttonAutoStart->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelToolsTrain::onButtonAutoStart, this);
    sizerAuto->Add(buttonAutoStart);
    buttonAutoStop = new wxButton(this, wxID_ANY, "Stop");
    buttonAutoStop->Disable();
    buttonAutoStop->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelToolsTrain::onButtonAutoStop, this);
    sizerAuto->Add(buttonAutoStop);

    wxBoxSizer* sizerManual; // TODO

    wxBoxSizer* sizerIntervals = new wxBoxSizer(wxVERTICAL);
    sizerIntervals->SetMinSize(120, -1);
    sizer->Add(sizerIntervals, wxSizerFlags().Border());

    sizerIntervals->Add(new wxStaticText(this, wxID_ANY, "Capture Interval"));
    textIntervalCapture = new wxStaticText(this, wxID_ANY, "");
    sizerIntervals->Add(textIntervalCapture);
    spinIntervalCapture = new wxSpinCtrl(this);
    spinIntervalCapture->SetRange(0, 1000000);
    spinIntervalCapture->SetMinSize({64, -1});
    spinIntervalCapture->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsTrain::onSpinIntervalCapture, this);
    sizerIntervals->Add(spinIntervalCapture);

    sizerIntervals->Add(new wxStaticText(this, wxID_ANY, "Densify Interval"), wxSizerFlags().Border(wxUP));
    textIntervalDensify = new wxStaticText(this, wxID_ANY, "");
    sizerIntervals->Add(textIntervalDensify);
    spinIntervalDensify = new wxSpinCtrl(this);
    spinIntervalDensify->SetRange(0, 1000000);
    spinIntervalDensify->SetMinSize({64, -1});
    spinIntervalDensify->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsTrain::onSpinIntervalDensify, this);
    sizerIntervals->Add(spinIntervalDensify);

    SetSizerAndFit(sizer);
}

void UiPanelToolsTrain::refreshProject() {
    spinIntervalCapture->SetValue(getProject().intervalCapture);
    spinIntervalDensify->SetValue(getProject().intervalDensify);
    refreshText();
}

void UiPanelToolsTrain::refreshText() {
    if (getProject().intervalCapture > 0) {
        textIntervalCapture->SetLabel(std::format(" > in {} iteration(s)",
                                                  getProject().intervalCapture - (getProject().iterations % getProject().intervalCapture)));
    } else textIntervalCapture->SetLabel(" > never");
    if (getProject().intervalDensify > 0) {
        textIntervalDensify->SetLabel(std::format(" > in {} iteration(s)",
                                                  getProject().intervalDensify - (getProject().iterations % getProject().intervalDensify)));
    } else textIntervalDensify->SetLabel(" > never");
}

void UiPanelToolsTrain::onButtonAutoStart(wxCommandEvent& event) {
    getFrame().autoTraining = true;
    buttonAutoStart->Disable();
    buttonAutoStop->Enable();
}

void UiPanelToolsTrain::onButtonAutoStop(wxCommandEvent& event) {
    getFrame().autoTraining = false;
    buttonAutoStart->Enable();
    buttonAutoStop->Disable();
}

void UiPanelToolsTrain::onSpinIntervalCapture(wxSpinEvent& event) {
    getProject().intervalCapture = spinIntervalCapture->GetValue();
    refreshText();
}

void UiPanelToolsTrain::onSpinIntervalDensify(wxSpinEvent& event) {
    getProject().intervalDensify = spinIntervalDensify->GetValue();
    refreshText();
}
