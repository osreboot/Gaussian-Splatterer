#include "UiPanelParamsLr.h"

Project& UiPanelParamsLr::getProject() const {
    return *frame.project;
}

UiPanelParamsLr::UiPanelParamsLr(wxWindow* parent, UiFrame& frame) : wxPanel(parent), frame(frame) {
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);

    sizer->Add(new wxStaticText(this, wxID_ANY, "Location"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinLrLocation = new wxSpinCtrlDouble(this, LR_LOCATION);
    spinLrLocation->SetRange(0.0f, 10.0f);
    spinLrLocation->SetDigits(8);
    spinLrLocation->SetIncrement(0.0000001);
    spinLrLocation->SetMinSize({96, -1});
    sizer->Add(spinLrLocation, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    sizer->Add(new wxStaticText(this, wxID_ANY, "Color (SHs)"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinLrSh = new wxSpinCtrlDouble(this, LR_SH);
    spinLrSh->SetRange(0.0f, 10.0f);
    spinLrSh->SetDigits(8);
    spinLrSh->SetIncrement(0.0000001);
    spinLrSh->SetMinSize({96, -1});
    sizer->Add(spinLrSh, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    sizer->Add(new wxStaticText(this, wxID_ANY, "Scale"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinLrScale = new wxSpinCtrlDouble(this, LR_SCALE);
    spinLrScale->SetRange(0.0f, 10.0f);
    spinLrScale->SetDigits(8);
    spinLrScale->SetIncrement(0.0000001);
    spinLrScale->SetMinSize({96, -1});
    sizer->Add(spinLrScale, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    sizer->Add(new wxStaticText(this, wxID_ANY, "Opacity"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinLrOpacity = new wxSpinCtrlDouble(this, LR_OPACITY);
    spinLrOpacity->SetRange(0.0f, 10.0f);
    spinLrOpacity->SetDigits(8);
    spinLrOpacity->SetIncrement(0.0000001);
    spinLrOpacity->SetMinSize({96, -1});
    sizer->Add(spinLrOpacity, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    sizer->Add(new wxStaticText(this, wxID_ANY, "Rotation"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinLrRotation = new wxSpinCtrlDouble(this, LR_ROTATION);
    spinLrRotation->SetRange(0.0f, 10.0f);
    spinLrRotation->SetDigits(8);
    spinLrRotation->SetIncrement(0.0000001);
    spinLrRotation->SetMinSize({96, -1});
    sizer->Add(spinLrRotation, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelParamsLr::onSpinParameter, this);

    SetSizerAndFit(sizer);
}

void UiPanelParamsLr::refreshProject() {
    spinLrLocation->SetValue(getProject().lrLocation);
    spinLrSh->SetValue(getProject().lrSh);
    spinLrScale->SetValue(getProject().lrScale);
    spinLrOpacity->SetValue(getProject().lrOpacity);
    spinLrRotation->SetValue(getProject().lrRotation);
}

void UiPanelParamsLr::onSpinParameter(wxSpinDoubleEvent& event) {
    switch(event.GetId()) {
        case ParamIds::LR_LOCATION: getProject().lrLocation = (float)event.GetValue(); break;
        case ParamIds::LR_SH: getProject().lrSh = (float)event.GetValue(); break;
        case ParamIds::LR_SCALE: getProject().lrScale = (float)event.GetValue(); break;
        case ParamIds::LR_OPACITY: getProject().lrOpacity = (float)event.GetValue(); break;
        case ParamIds::LR_ROTATION: getProject().lrRotation = (float)event.GetValue(); break;
        default: break;
    }
}
