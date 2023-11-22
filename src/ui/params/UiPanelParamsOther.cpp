#include "UiPanelParamsOther.h"

UiFrame& UiPanelParamsOther::getFrame() const {
    return *dynamic_cast<UiFrame*>(GetParent()->GetParent()->GetParent());
}

Project& UiPanelParamsOther::getProject() const {
    return *getFrame().project;
}

UiPanelParamsOther::UiPanelParamsOther(wxWindow* parent) : wxPanel(parent) {
    sizer = new wxStaticBoxSizer(wxVERTICAL, this, "Other Parameters");

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
