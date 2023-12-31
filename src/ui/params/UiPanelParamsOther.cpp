#include "UiPanelParamsOther.h"

Project& UiPanelParamsOther::getProject() const {
    return *frame.project;
}

UiPanelParamsOther::UiPanelParamsOther(wxWindow* parent, UiFrame& frame) : wxPanel(parent), frame(frame) {
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);

    sizer->Add(new wxStaticText(this, wxID_ANY, "Max Splat Size"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinParamSizeMax = new wxSpinCtrlDouble(this, P_SIZE_MAX);
    spinParamSizeMax->SetToolTip("Splat scale axes will be limited to this maximum value. Applied every iteration.");
    spinParamSizeMax->SetRange(0.0f, 100000.0f);
    spinParamSizeMax->SetDigits(6);
    spinParamSizeMax->SetIncrement(0.0001);
    spinParamSizeMax->SetMinSize({96, -1});
    sizer->Add(spinParamSizeMax, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelParamsOther::onSpinParameter, this);

    SetSizerAndFit(sizer);
}

void UiPanelParamsOther::refreshProject() {
    spinParamSizeMax->SetValue(getProject().paramScaleMax);
}

void UiPanelParamsOther::onSpinParameter(wxSpinDoubleEvent& event) {
    switch(event.GetId()) {
        case ParamIds::P_SIZE_MAX: getProject().paramScaleMax = (float)event.GetValue(); break;
        default: break;
    }
}
