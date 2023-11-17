#include "UiPanelToolsTrain.h"

UiPanelToolsTrain::UiPanelToolsTrain(wxWindow* parent) : wxPanel(parent) {
    sizer = new wxStaticBoxSizer(wxHORIZONTAL, this, "3. Train Splats");

    sizerAuto = new wxBoxSizer(wxVERTICAL);
    sizer->Add(sizerAuto, wxSizerFlags().Border());

    sizerAuto->Add(new wxStaticText(this, wxID_ANY, "Auto Train"));
    buttonAutoStart = new wxButton(this, wxID_ANY, "Start");
    // TODO event
    sizerAuto->Add(buttonAutoStart);
    buttonAutoStop = new wxButton(this, wxID_ANY, "Stop");
    // TODO event
    sizerAuto->Add(buttonAutoStop);

    SetSizerAndFit(sizer);
}
